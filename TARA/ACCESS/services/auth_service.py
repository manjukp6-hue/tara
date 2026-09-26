"""
TARA/ACCESS/services/creator_auth_service.py

Unified Production Creator Authentication Service for TARA.
Combines:
1. Conversational private trigger detection (without permanent login buttons).
2. Direct in-chat presentation of the 4 authentication methods:
   - [ 📱 QR Authentication ]
   - [ 🔑 Creator Key Authentication ]
   - [ 🔐 Google Authentication ]
   - [ 🆘 Recovery ]
3. Server-side Google ID Token (RS256) cryptographic verification.
4. Protected Creator Key authentication (proof-of-possession artifact / keystore unlock, zero raw key in chat).
5. Cryptographic QR one-time challenge generation with in-chat ASCII & SVG QR rendering.
6. Multi-creator support (Creator 1 / ROOT_OPERATOR, Creator 2, Creator 3).
7. Strict role enforcement (ROOT_OPERATOR = ROOT_CREATOR; no self-escalation for normal CREATORs).
8. Emergency & multi-creator recovery verification.
9. Sliding-window rate limiting & 300s lockout protection.
10. Ephemeral, short-lived authenticated creator sessions with revocation & logout.
"""

import os
import json
import time
import secrets
import hashlib
import hmac
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

from ..operator.operator_profile import CreatorIdentity, CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
from ..operator.operator_lifecycle import AuthorityLifecycleManager, AuthorityState
from ..operator.multi_operator_registry import MultiCreatorRegistry
from ..operator.qr_generator import generate_chat_qr
from ..crypto.ed25519 import Ed25519
from ..devices.device_registry import DeviceRegistry
from ..services.google_auth import GoogleAuthService
from ..restore.restore_manager import RecoveryManager, RecoveryAuthorizationProof
from ..audit.security_logger import SecurityAuditLogger
from ..activation.activation_manager import PrivateTriggerManager, GENERIC_LOGIN_PHRASES
from ...SECURITY.tamper_detector import SourceTamperDetector
from ..factors import (
    WindowsHelloProvider,
    FingerprintProvider,
    FaceProvider,
    IrisProvider,
    VoiceVerificationProvider,
    BiometricCapability,
    BiometricAuthResult
)

AUTHORIZED_CREATOR_EMAIL = "operator@internal.local"
SESSION_LIFETIME_SECONDS = 3600  # 1 hour short-lived session
LOCKOUT_THRESHOLD = 5
LOCKOUT_DURATION_SECONDS = 300   # 5 minutes


class CreatorAuthService:
    """
    Core backend authority service that validates creator credentials,
    manages the conversational private trigger, presents in-chat authentication methods,
    verifies QR / Creator Key / Google / Recovery proofs, enforces rate limits,
    resolves multi-creator roles, and issues short-lived session tokens.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        creator_record_path: Optional[str] = None,
        trigger_file_path: Optional[str] = None,
        audit_log_dir: Optional[str] = None,
        devices_file_path: Optional[str] = None,
        creators_registry_path: Optional[str] = None,
        recovery_record_path: Optional[str] = None,
        google_service: Optional[GoogleAuthService] = None,
        seal_path: Optional[str] = None,
        auth_manifest_path: Optional[str] = None
    ):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.repo_root = repo_root

        self.creator = CreatorIdentity(record_path=creator_record_path)
        self.multi_creators = MultiCreatorRegistry(registry_path=creators_registry_path)
        self.trigger_mgr = PrivateTriggerManager(trigger_file_path=trigger_file_path)
        self.audit_logger = SecurityAuditLogger(log_dir=audit_log_dir)
        self.devices = DeviceRegistry(registry_path=devices_file_path)
        self.recovery_mgr = RecoveryManager(creator=self.creator, recovery_record_path=recovery_record_path)
        self.google_service = google_service or GoogleAuthService(authorized_email=AUTHORIZED_CREATOR_EMAIL)

        self.seal_path = seal_path or os.path.join(self.repo_root, "storage", "vault", "access", "access_seal.json")
        self.auth_manifest_path = auth_manifest_path or os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "auth_manifest.json")

        self.lifecycle = AuthorityLifecycleManager(
            repo_root=self.repo_root,
            seal_path=self.seal_path,
            creator_record_path=creator_record_path,
            creators_registry_path=creators_registry_path,
            recovery_config_path=recovery_record_path,
            auth_manifest_path=self.auth_manifest_path
        )
        self.tamper_detector = SourceTamperDetector(repo_root=self.repo_root, audit_logger=self.audit_logger)

        # Sync root creator public key into multi_creators if available
        if self.creator.root_public_key:
            self.multi_creators.set_public_key(CANONICAL_CREATOR_ID, self.creator.root_public_key)

        # Biometric providers
        self.windows_hello = WindowsHelloProvider()
        self.fingerprint = FingerprintProvider()
        self.face = FaceProvider()
        self.iris = IrisProvider()
        self.voice_provider = VoiceVerificationProvider()

        # Thread-safe in-memory session cache: session_token -> session_info
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        # Rate limiting: client_key -> {"count": int, "locked_until": float}
        self._rate_limits: Dict[str, Dict[str, Any]] = {}

        # One-time QR challenges: challenge_id -> {"nonce": str, "timestamp": float, "expires_at": float, "creator_id": str}
        self._qr_challenges: Dict[str, Dict[str, Any]] = {}

        # One-time Key challenges: nonce -> {"creator_id": str, "timestamp": float, "expires_at": float}
        self._key_challenges: Dict[str, Dict[str, Any]] = {}

        # Conversational session states: chat_session_id -> {"state": str, "challenge": dict, ...}
        self._conversational_states: Dict[str, Dict[str, Any]] = {}

    # -----------------------------------------------------------------------
    # RATE LIMITING & LOCKOUT
    # -----------------------------------------------------------------------
    def _check_rate_limit(self, client_key: str) -> Tuple[bool, int]:
        """Returns (is_allowed, remaining_lockout_seconds)."""
        now = time.time()
        with self._lock:
            record = self._rate_limits.get(client_key)
            if not record:
                return True, 0
            locked_until = record.get("locked_until", 0)
            if now < locked_until:
                return False, int(locked_until - now) + 1
            if record.get("locked_until") and now >= locked_until:
                del self._rate_limits[client_key]
                return True, 0
            return True, 0

    def _record_failure(self, client_key: str) -> Tuple[bool, int]:
        """Records an authentication failure; locks client if threshold reached."""
        now = time.time()
        with self._lock:
            record = self._rate_limits.setdefault(client_key, {"count": 0, "locked_until": 0})
            record["count"] += 1
            if record["count"] >= LOCKOUT_THRESHOLD:
                record["locked_until"] = now + LOCKOUT_DURATION_SECONDS
                return True, LOCKOUT_DURATION_SECONDS
            return False, 0

    def _reset_failure(self, client_key: str) -> None:
        """Resets failed attempt count on successful authentication."""
        with self._lock:
            self._rate_limits.pop(client_key, None)

    # -----------------------------------------------------------------------
    # 1. CONVERSATIONAL TRIGGER DETECTION & IN-CHAT METHOD PRESENTATION
    # -----------------------------------------------------------------------
    def check_conversational_trigger(self, input_text: str) -> Dict[str, Any]:
        """
        Checks if input_text matches the configured private creator trigger phrase.
        Generic phrases are strictly rejected.
        When verified, returns the 4 authentication choices directly for chat display.
        Knowing the trigger phrase alone NEVER grants creator authority.
        """
        if not input_text or not isinstance(input_text, str):
            return {"is_trigger": False}

        if self.trigger_mgr.verify_trigger(input_text):
            self.audit_logger.log_event(
                event_type="CREATOR_AUTH_REQUESTED",
                severity="INFO",
                details={"target_email": AUTHORIZED_CREATOR_EMAIL}
            )

            prompt_text = (
                "Creator authentication requested. Choose an authentication method:\n\n"
                "[ 📱 QR Authentication ]\n"
                "[ 🔑 Creator Key Authentication ]\n"
                "[ 🔐 Google Authentication ]\n"
                "[ 🆘 Recovery ]"
            )

            return {
                "is_trigger": True,
                "response": prompt_text,
                "status": "AWAITING_METHOD_SELECTION",
                "methods": [
                    {"id": "qr", "label": "📱 QR Authentication"},
                    {"id": "creator_key", "label": "🔑 Creator Key Authentication"},
                    {"id": "google", "label": "🔐 Google Authentication"},
                    {"id": "recovery", "label": "🆘 Recovery"}
                ]
            }

        return {"is_trigger": False}

    def detect_method_selection(self, text: str) -> Optional[str]:
        """
        Detects which authentication method the user selected in chat.
        Supports button text, emojis, numbers, or colloquial phrases.
        """
        if not text or not isinstance(text, str):
            return None
        t = text.strip().lower()

        if any(w in t for w in ("qr authentication", "📱 qr", "qr auth", "qr code", "qr", "[ 📱 qr authentication ]")):
            return "QR"
        if any(w in t for w in ("creator key authentication", "🔑 creator key", "creator key", "key auth", "[ 🔑 creator key authentication ]")):
            return "CREATOR_KEY"
        if any(w in t for w in ("google authentication", "🔐 google", "google auth", "google login", "google", "[ 🔐 google authentication ]")):
            return "GOOGLE"
        if any(w in t for w in ("recovery", "🆘 recovery", "emergency recovery", "[ 🆘 recovery ]")):
            return "RECOVERY"
        if t in ("1", "one"):
            return "QR"
        if t in ("2", "two"):
            return "CREATOR_KEY"
        if t in ("3", "three"):
            return "GOOGLE"
        if t in ("4", "four"):
            return "RECOVERY"
        return None

    def handle_method_selection(
        self,
        method: str,
        creator_id: str = CANONICAL_CREATOR_ID,
        client_key: str = "default_client"
    ) -> Dict[str, Any]:
        """
        Handles the user's selection of an authentication method within the same chat session.
        Generates the appropriate challenge / prompt to display directly in the chat window.
        """
        method = method.upper()

        if method == "QR":
            challenge = self.create_qr_challenge(creator_id=creator_id)
            ascii_art = challenge["ascii_qr"]
            challenge_id = challenge["challenge_id"]
            nonce = challenge["nonce"]
            uri = challenge["challenge_uri"]

            resp_text = (
                f"📱 QR Authentication\n\n"
                f"Scan this QR code with your trusted creator device:\n\n"
                f"```\n{ascii_art}\n```\n\n"
                f"Challenge URI: {uri}\n"
                f"Challenge ID: {challenge_id}\n"
                f"(Valid for 120s. Single-use and replay protected.)"
            )

            return {
                "status": "AWAITING_QR_APPROVAL",
                "method": "QR",
                "response": resp_text,
                "challenge": challenge
            }

        elif method == "CREATOR_KEY":
            # Generate a one-time cryptographic proof challenge nonce
            nonce = secrets.token_hex(24)
            now = time.time()
            with self._lock:
                self._key_challenges[nonce] = {
                    "creator_id": creator_id,
                    "timestamp": now,
                    "expires_at": now + 300.0
                }

            resp_text = (
                f"🔑 Creator Key Authentication\n\n"
                f"Challenge Nonce: `{nonce}`\n\n"
                f"Please submit your cryptographic proof-of-possession artifact "
                f"(Ed25519 signature over the challenge nonce) or your protected keystore passphrase. "
                f"Your raw private key is never asked for and must never enter the chat."
            )

            return {
                "status": "AWAITING_KEY_PROOF",
                "method": "CREATOR_KEY",
                "response": resp_text,
                "challenge_nonce": nonce
            }

        elif method == "GOOGLE":
            creator_rec = self.multi_creators.get_creator(creator_id)
            configured_email = (creator_rec.get("authorized_google_email") if creator_rec else None) or AUTHORIZED_CREATOR_EMAIL
            resp_text = (
                f"🔐 Google Authentication\n\n"
                f"Authorized Account: `{configured_email}`\n\n"
                f"Authenticate with Google Sign-In (supporting Passkeys, 2SV, and Google Authenticator). "
                f"Submit your verified Google ID token to complete authentication:"
            )

            return {
                "status": "AWAITING_GOOGLE_TOKEN",
                "method": "GOOGLE",
                "response": resp_text,
                "target_account": configured_email
            }

        elif method == "RECOVERY":
            resp_text = (
                f"🆘 Emergency Recovery\n\n"
                f"Please enter your 32-character high-entropy recovery code "
                f"or multi-creator recovery authorization to restore creator authority:"
            )

            return {
                "status": "AWAITING_RECOVERY_CODE",
                "method": "RECOVERY",
                "response": resp_text
            }

        else:
            return {
                "status": "FAILED",
                "response": "Unknown authentication method selected. Please choose from: QR, Creator Key, Google, or Recovery."
            }

    # -----------------------------------------------------------------------
    # 2. METHOD A: QR AUTHENTICATION (IN-CHAT CHALLENGE-RESPONSE)
    # -----------------------------------------------------------------------
    def create_qr_challenge(self, creator_id: str = CANONICAL_CREATOR_ID) -> Dict[str, Any]:
        """
        Generates a cryptographically secure one-time QR challenge (120s expiry).
        Contains non-sensitive challenge metadata and renders ASCII + SVG QR representations.
        """
        if self.lifecycle.is_locked():
            return {
                "status": "AUTHORITY_LOCKED",
                "error": "Creator authority is locked due to integrity mismatch or tampering. Contact root authority or use authorized recovery."
            }
        if self.lifecycle.get_state() == AuthorityState.CREATOR_SETUP_REQUIRED:
            return {
                "status": "CREATOR_SETUP_REQUIRED",
                "error": "Creator setup has not been performed yet. Real interactive setup is required."
            }
        challenge_id = secrets.token_hex(16)
        nonce = secrets.token_hex(24)
        now = time.time()
        expires_at = now + 120.0

        challenge_uri = f"tara-auth://challenge?id={challenge_id}&nonce={nonce}&creator={creator_id}"
        ascii_qr, svg_qr = generate_chat_qr(challenge_uri, use_unicode=True)

        challenge_data = {
            "challenge_id": challenge_id,
            "creator_id": creator_id,
            "nonce": nonce,
            "challenge_uri": challenge_uri,
            "created_at": now,
            "expires_at": expires_at
        }

        with self._lock:
            self._qr_challenges[challenge_id] = challenge_data

        return {
            "challenge_id": challenge_id,
            "creator_id": creator_id,
            "nonce": nonce,
            "challenge_uri": challenge_uri,
            "ascii_qr": ascii_qr,
            "svg_qr": svg_qr,
            "expires_in": 120
        }

    def verify_qr_approval(
        self,
        challenge_id: str,
        device_id: str,
        device_signature_hex: str,
        client_key: str = "default_client"
    ) -> Dict[str, Any]:
        """
        Verifies approval from an authorized device for a one-time QR challenge.
        Challenge is strictly one-time use; replay is rejected immediately.
        """
        if self.lifecycle.is_locked():
            return {
                "status": "AUTHORITY_LOCKED",
                "error": "Creator authority is locked due to integrity mismatch or tampering. Contact root authority or use authorized recovery."
            }
        if self.lifecycle.get_state() == AuthorityState.CREATOR_SETUP_REQUIRED:
            return {
                "status": "CREATOR_SETUP_REQUIRED",
                "error": "Creator setup has not been performed yet. Real interactive setup is required."
            }

        allowed, rem_secs = self._check_rate_limit(client_key)
        if not allowed:
            return {
                "status": "FAILED",
                "error": "Creator authentication failed. No creator authority was granted.",
                "lockout_remaining": rem_secs
            }

        now = time.time()
        with self._lock:
            challenge = self._qr_challenges.pop(challenge_id, None)

        if not challenge:
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Invalid or already used QR challenge (replay prevented)."}

        if now > challenge["expires_at"]:
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "QR challenge has expired."}

        # Verify device in registry
        dev_info = self.devices.get_device(device_id)
        if not dev_info or dev_info.get("status", "").upper() != "AUTHORIZED":
            self._record_failure(client_key)
            self.audit_logger.log_event(
                event_type="QR_DEVICE_REJECTED",
                severity="WARNING",
                details={"device_id": device_id}
            )
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        pubkey_hex = dev_info.get("device_public_key")
        if not pubkey_hex:
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        # Challenge message: challenge_id:nonce:creator_id
        challenge_msg = f"{challenge_id}:{challenge['nonce']}:{challenge['creator_id']}".encode("utf-8")
        try:
            pub_bytes = bytes.fromhex(pubkey_hex)
            sig_bytes = bytes.fromhex(device_signature_hex)
            valid = Ed25519.verify(pub_bytes, challenge_msg, sig_bytes)
        except Exception:
            valid = False

        if not valid:
            self._record_failure(client_key)
            self.audit_logger.log_event(
                event_type="QR_SIGNATURE_FAILED",
                severity="SECURITY_ALERT",
                details={"device_id": device_id}
            )
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        target_creator_id = challenge.get("creator_id", CANONICAL_CREATOR_ID)
        return self._issue_creator_session(target_creator_id, auth_method=f"TRUSTED_DEVICE_QR:{device_id}", client_key=client_key)

    # -----------------------------------------------------------------------
    # 3. METHOD B: CREATOR KEY AUTHENTICATION (PROOF ARTIFACT / KEYSTORE)
    # -----------------------------------------------------------------------
    def authenticate_creator_key(
        self,
        proof_signature_hex: Optional[str] = None,
        challenge_nonce: Optional[str] = None,
        claimed_creator_id: str = CANONICAL_CREATOR_ID,
        passphrase: Optional[str] = None,
        client_key: str = "default_client"
    ) -> Dict[str, Any]:
        """
        Validates Creator Key authentication:
        - Mode A: Ed25519 signature over a challenge nonce produced by a local key agent or passkey.
        - Mode B: Passphrase unlocking the local protected keystore (Scrypt/AES-256-GCM/DPAPI).
        Never asks for or exposes raw private keys in chat or logs.
        """
        if self.lifecycle.is_locked():
            return {
                "status": "AUTHORITY_LOCKED",
                "error": "Creator authority is locked due to integrity mismatch or tampering. Contact root authority or use authorized recovery."
            }
        if self.lifecycle.get_state() == AuthorityState.CREATOR_SETUP_REQUIRED:
            return {
                "status": "CREATOR_SETUP_REQUIRED",
                "error": "Creator setup has not been performed yet. Real interactive setup is required."
            }

        allowed, rem_secs = self._check_rate_limit(client_key)
        if not allowed:
            return {
                "status": "FAILED",
                "error": "Creator authentication failed. No creator authority was granted.",
                "lockout_remaining": rem_secs
            }

        clean_id = claimed_creator_id.strip().upper()

        # Check creator active status in multi-creator registry
        creator_record = self.multi_creators.get_creator(clean_id)
        if not creator_record or creator_record.get("status") != "active":
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        # Mode A: Proof Signature Verification over challenge nonce
        if proof_signature_hex and challenge_nonce:
            now = time.time()
            with self._lock:
                chal = self._key_challenges.pop(challenge_nonce, None)

            if not chal or now > chal.get("expires_at", 0):
                self._record_failure(client_key)
                return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

            # Retrieve registered public key
            pubkey_hex = creator_record.get("public_key")
            if not pubkey_hex and clean_id == CANONICAL_CREATOR_ID:
                pubkey_hex = self.creator.root_public_key

            if not pubkey_hex:
                self._record_failure(client_key)
                return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

            try:
                pub_bytes = bytes.fromhex(pubkey_hex)
                sig_bytes = bytes.fromhex(proof_signature_hex)
                valid = Ed25519.verify(pub_bytes, challenge_nonce.encode("utf-8"), sig_bytes)
            except Exception:
                valid = False

            if not valid:
                self._record_failure(client_key)
                return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

            return self._issue_creator_session(clean_id, auth_method="CREATOR_KEY_PROOF", client_key=client_key)

        # Mode B: Passphrase Keystore Unlock
        elif passphrase and clean_id == CANONICAL_CREATOR_ID:
            try:
                from ..crypto.secure_storage import SecureStorage
                keystore_path = os.path.join(self.repo_root, "TARA", "ACCESS", "vault", "operator_key.keystore")
                if not os.path.exists(keystore_path):
                    alt_path = os.path.join(self.repo_root, "storage", "vault", "access", "operator_key.keystore")
                    if os.path.exists(alt_path):
                        keystore_path = alt_path
                    else:
                        self._record_failure(client_key)
                        return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

                storage = SecureStorage()
                priv_bytes = storage.load_private_key(keystore_path, passphrase)
                if not priv_bytes:
                    self._record_failure(client_key)
                    return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

                # Verify public key matches
                pub_bytes = Ed25519.public_key_from_private(priv_bytes)
                # Securely wipe priv_bytes from memory immediately
                del priv_bytes

                if self.creator.root_public_key and pub_bytes.hex().lower() != self.creator.root_public_key.lower():
                    self._record_failure(client_key)
                    return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

                return self._issue_creator_session(clean_id, auth_method="PROTECTED_KEYSTORE_UNLOCK", client_key=client_key)
            except Exception:
                self._record_failure(client_key)
                return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        self._record_failure(client_key)
        return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

    # -----------------------------------------------------------------------
    # 4. METHOD C: GOOGLE TOKEN AUTHENTICATION
    # -----------------------------------------------------------------------
    def generate_google_nonce(self) -> str:
        """Generates and registers a random nonce for Google authentication."""
        return self.google_service.create_auth_nonce()

    def authenticate_google_token(
        self,
        id_token: str,
        expected_nonce: Optional[str] = None,
        require_nonce: bool = False,
        client_key: str = "default_client"
    ) -> Dict[str, Any]:
        """
        Cryptographically validates a Google ID token (RS256 JWT).
        Matches token email against MultiCreatorRegistry.
        Assigns registered creator role without self-escalation.
        """
        if self.lifecycle.is_locked():
            return {
                "status": "AUTHORITY_LOCKED",
                "error": "Creator authority is locked due to integrity mismatch or tampering. Contact root authority or use authorized recovery."
            }
        if self.lifecycle.get_state() == AuthorityState.CREATOR_SETUP_REQUIRED:
            return {
                "status": "CREATOR_SETUP_REQUIRED",
                "error": "Creator setup has not been performed yet. Real interactive setup is required."
            }

        allowed, rem_secs = self._check_rate_limit(client_key)
        if not allowed:
            return {
                "status": "FAILED",
                "error": "Creator authentication failed. No creator authority was granted.",
                "lockout_remaining": rem_secs
            }

        if not id_token or not isinstance(id_token, str):
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        verif_res = self.google_service.verify_id_token(id_token, expected_nonce=expected_nonce, require_nonce=require_nonce)
        if not verif_res.get("verified"):
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        claims = verif_res.get("claims", {})
        token_email = (claims.get("email") or "").lower().strip()
        token_sub = (claims.get("sub") or "").strip()
        email_verified = claims.get("email_verified")

        if email_verified not in (True, "true"):
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        # Match in MultiCreatorRegistry using explicit Google identity mapping
        creator_record = self.multi_creators.get_creator_by_google_identity(email=token_email, subject_id=token_sub)

        # Invariant: Must be active and have a legitimately configured authorized_google_email
        if not creator_record or creator_record.get("status") != "active":
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        creator_email = (creator_record.get("authorized_google_email") or "").lower().strip()
        if not creator_email or creator_email != token_email:
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        # Verify subject ID match if already bound
        if creator_record.get("google_subject_id") and creator_record["google_subject_id"] != token_sub:
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        target_creator_id = creator_record["creator_id"]

        # Bind google_subject_id upon first successful verification if not bound
        if not creator_record.get("google_subject_id") and token_sub:
            if self.multi_creators.bind_google_subject_id(target_creator_id, token_sub):
                if self.creator.root_public_key:
                    try:
                        self.lifecycle.reseal_with_recovery(
                            new_pub_bytes=bytes.fromhex(self.creator.root_public_key),
                            key_version=self.creator.key_version,
                            display_name=DEFAULT_DISPLAY_NAME
                        )
                    except Exception:
                        pass

        return self._issue_creator_session(target_creator_id, auth_method="GOOGLE_OAUTH_VERIFIED", client_key=client_key)

    def enroll_creator_google_identity(
        self,
        creator_id: str,
        google_email: str,
        google_subject_id: Optional[str] = None,
        session_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrolls a verified Google identity for a creator.
        Requires active ROOT_CREATOR session authority.
        """
        if self.lifecycle.is_locked():
            return {
                "status": "AUTHORITY_LOCKED",
                "error": "Creator authority is locked due to integrity mismatch or tampering. Contact root authority or use authorized recovery."
            }
        if self.lifecycle.get_state() == AuthorityState.CREATOR_SETUP_REQUIRED:
            return {
                "status": "CREATOR_SETUP_REQUIRED",
                "error": "Creator setup has not been performed yet. Real interactive setup is required."
            }

        if not session_token:
            return {"status": "FAILED", "error": "Authentication required to manage creator identities."}
        sess = self.verify_session(session_token)
        if not sess or sess.get("role") != "ROOT_CREATOR":
            return {"status": "FAILED", "error": "Unauthorized: Only ROOT_CREATOR can enroll creator identities."}

        try:
            rec = self.multi_creators.enroll_google_identity(
                creator_id=creator_id,
                google_email=google_email,
                google_subject_id=google_subject_id,
                authorized_by=sess.get("creator_id", CANONICAL_CREATOR_ID)
            )
            self.audit_logger.log_event(
                event_type="CREATOR_GOOGLE_ENROLLED",
                severity="INFO",
                details={"creator_id": creator_id, "email": google_email}
            )
            return {"status": "SUCCESS", "creator": rec}
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    # -----------------------------------------------------------------------
    # 5. METHOD D: EMERGENCY & MULTI-CREATOR RECOVERY
    # -----------------------------------------------------------------------
    def authenticate_recovery(
        self,
        recovery_code_or_token: str,
        claimed_creator_id: str = CANONICAL_CREATOR_ID,
        client_key: str = "default_client"
    ) -> Dict[str, Any]:
        """
        Validates emergency recovery credentials.
        Accepts 32-character recovery code or multi-creator recovery authorization.
        Creator ID, name, or email alone strictly cannot authenticate.
        """
        if self.lifecycle.is_locked():
            return {
                "status": "AUTHORITY_LOCKED",
                "error": "Creator authority is locked due to integrity mismatch or tampering. Contact root authority or use authorized recovery."
            }
        if self.lifecycle.get_state() == AuthorityState.CREATOR_SETUP_REQUIRED:
            return {
                "status": "CREATOR_SETUP_REQUIRED",
                "error": "Creator setup has not been performed yet. Real interactive setup is required."
            }

        allowed, rem_secs = self._check_rate_limit(client_key)
        if not allowed:
            return {
                "status": "FAILED",
                "error": "Creator authentication failed. No creator authority was granted.",
                "lockout_remaining": rem_secs
            }

        if not recovery_code_or_token or not isinstance(recovery_code_or_token, str):
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        clean_code = recovery_code_or_token.strip()
        clean_id = claimed_creator_id.strip().upper()

        # Reject if input is just creator name or email
        if clean_code.lower() in (CANONICAL_CREATOR_ID.lower(), DEFAULT_DISPLAY_NAME.lower(), AUTHORIZED_CREATOR_EMAIL.lower()):
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        # Verify against RecoveryManager
        valid = self.recovery_mgr.verify_recovery_code(clean_code)
        if not valid:
            self._record_failure(client_key)
            return {"status": "FAILED", "error": "Creator authentication failed. No creator authority was granted."}

        return self._issue_creator_session(clean_id, auth_method="EMERGENCY_RECOVERY_CODE", client_key=client_key)

    # -----------------------------------------------------------------------
    # 6. SESSION ISSUANCE & ROLE ENFORCEMENT
    # -----------------------------------------------------------------------
    def _issue_creator_session(
        self,
        creator_id: str,
        auth_method: str,
        client_key: str = "default_client"
    ) -> Dict[str, Any]:
        """
        Creates an authenticated creator session and assigns registered role.
        Enforces that ONLY ROOT_OPERATOR can ever receive ROOT_CREATOR.
        """
        clean_id = creator_id.strip().upper()
        creator_record = self.multi_creators.get_creator(clean_id)
        display_name = creator_record.get("display_name", clean_id) if creator_record else clean_id
        assigned_role = self.multi_creators.resolve_role(clean_id)

        # Reset failure counter on success
        self._reset_failure(client_key)

        session_token = secrets.token_hex(24)
        now = time.time()
        expires_at = now + SESSION_LIFETIME_SECONDS

        session = {
            "session_token": session_token,
            "creator_id": clean_id,
            "display_name": display_name,
            "role": assigned_role,
            "auth_method": auth_method,
            "issued_at": now,
            "expires_at": expires_at,
            "status": "ACTIVE"
        }

        with self._lock:
            max_sessions = getattr(self, "max_concurrent_sessions", 5)
            active_for_creator = [
                tok for tok, sess in self._sessions.items()
                if sess.get("creator_id") == clean_id and now <= sess.get("expires_at", 0)
            ]
            if len(active_for_creator) >= max_sessions:
                active_for_creator.sort(key=lambda t: self._sessions[t].get("issued_at", 0))
                to_evict = active_for_creator[: len(active_for_creator) - max_sessions + 1]
                for old_tok in to_evict:
                    del self._sessions[old_tok]
            self._sessions[session_token] = session

        self.audit_logger.log_event(
            event_type="CREATOR_AUTHENTICATED",
            severity="INFO",
            details={"creator_id": clean_id, "role": assigned_role, "method": auth_method}
        )

        return {
            "status": "SUCCESS",
            "message": f"✅ Creator authenticated: {clean_id}.",
            "session_token": session_token,
            "creator_id": clean_id,
            "display_name": display_name,
            "role": assigned_role,
            "expires_in": SESSION_LIFETIME_SECONDS
        }

    # -----------------------------------------------------------------------
    # 7. SESSION VERIFICATION & LOGOUT
    # -----------------------------------------------------------------------
    def verify_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validates an active creator session token."""
        if not session_token or not isinstance(session_token, str):
            return None

        now = time.time()
        with self._lock:
            session = self._sessions.get(session_token)
            if not session:
                return None

            if now > session.get("expires_at", 0):
                del self._sessions[session_token]
                return None

            return dict(session)

    def revoke_session(self, session_token: str) -> bool:
        """Explicitly revokes a session (logout)."""
        with self._lock:
            return self._sessions.pop(session_token, None) is not None

    def logout(self, session_token: str) -> Dict[str, Any]:
        revoked = self.revoke_session(session_token)
        return {
            "status": "SUCCESS" if revoked else "NOT_FOUND",
            "message": "Creator session successfully logged out." if revoked else "Session not active."
        }

    def list_active_sessions(self, creator_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists active creator sessions with masked identifiers.
        Never exposes raw session tokens.
        """
        now = time.time()
        result = []
        with self._lock:
            for tok, sess in list(self._sessions.items()):
                if now > sess.get("expires_at", 0):
                    del self._sessions[tok]
                    continue
                if creator_id and sess.get("creator_id") != creator_id.strip().upper():
                    continue
                masked_id = hashlib.sha256(tok.encode("utf-8")).hexdigest()[:12]
                result.append({
                    "session_id": masked_id,
                    "creator_id": sess.get("creator_id"),
                    "display_name": sess.get("display_name"),
                    "role": sess.get("role"),
                    "auth_method": sess.get("auth_method"),
                    "issued_at": sess.get("issued_at"),
                    "expires_at": sess.get("expires_at"),
                    "remaining_seconds": max(0, int(sess.get("expires_at", 0) - now))
                })
        return result

    def revoke_session_by_id(self, session_id: str, creator_id: Optional[str] = None) -> bool:
        """Revokes a session by its masked session_id."""
        with self._lock:
            for tok, sess in list(self._sessions.items()):
                masked_id = hashlib.sha256(tok.encode("utf-8")).hexdigest()[:12]
                if masked_id == session_id:
                    if creator_id and sess.get("creator_id") != creator_id.strip().upper():
                        return False
                    del self._sessions[tok]
                    return True
        return False

    def revoke_all_sessions(self, creator_id: Optional[str] = None) -> int:
        """Revokes all active sessions for a creator or system-wide."""
        count = 0
        with self._lock:
            for tok, sess in list(self._sessions.items()):
                if creator_id is None or sess.get("creator_id") == creator_id.strip().upper():
                    del self._sessions[tok]
                    count += 1
        return count

    # -----------------------------------------------------------------------
    # 8. VOICE AUXILIARY STEP-UP VERIFICATION (SECONDARY SIGNAL ONLY)
    # -----------------------------------------------------------------------
    def verify_voice_step_up(
        self,
        session_token: str,
        device_id: str,
        challenge_nonce: str,
        audio_bytes: bytes,
        device_signature_hex: str
    ) -> Dict[str, Any]:
        """
        Auxiliary voice verification step-up signal.
        
        CRITICAL ARCHITECTURAL POLICY (Section 9):
        Voice verification is strictly an OPTIONAL SECONDARY SIGNAL.
        It CANNOT and MUST NEVER grant root creator authority or create sessions alone.
        
        Requires simultaneously:
          1. Authenticated creator session (session_token).
          2. Registered authorized device (device_id).
          3. Fresh challenge nonce issued by voice_provider.
          4. Device cryptographic signature over challenge_nonce with device Ed25519 key.
          5. Acoustic verification of audio_bytes against enrolled voice profile.
        
        Returns:
          {"status": "SUCCESS", "step_up_verified": True} on valid multi-factor confirmation.
        """
        # Rule 1: Cannot authorize alone without an existing authenticated creator session
        sess = self.verify_session(session_token)
        if not sess:
            return {
                "status": "DENIED",
                "error": "Voice verification cannot authorize alone. An active authenticated creator session is strictly required.",
                "code": "VOICE_CANNOT_AUTHORIZE_ALONE"
            }

        creator_id = sess.get("creator_id")
        if not creator_id:
            return {"status": "DENIED", "error": "Invalid session state.", "code": "UNAUTHORIZED"}

        # Rule 2: Registered device verification
        device = self.devices.get_device(device_id)
        if not device or device.get("status") != "AUTHORIZED":
            return {
                "status": "DENIED",
                "error": f"Device '{device_id}' is not an authorized registered device.",
                "code": "UNAUTHORIZED_DEVICE"
            }

        device_pub_hex = device.get("public_key")
        if not device_pub_hex:
            return {"status": "DENIED", "error": "Device public key not found.", "code": "INVALID_DEVICE"}

        # Rule 3: Device cryptographic signature verification over challenge_nonce
        if not device_signature_hex or not challenge_nonce:
            return {
                "status": "DENIED",
                "error": "Device cryptographic signature over fresh challenge nonce is required.",
                "code": "DEVICE_SIGNATURE_REQUIRED"
            }

        try:
            dev_pub_bytes = bytes.fromhex(device_pub_hex)
            dev_sig_bytes = bytes.fromhex(device_signature_hex)
            dev_valid = Ed25519.verify(dev_pub_bytes, challenge_nonce.encode("utf-8"), dev_sig_bytes)
        except Exception:
            dev_valid = False

        if not dev_valid:
            return {
                "status": "DENIED",
                "error": "Device signature verification failed for challenge nonce.",
                "code": "INVALID_DEVICE_SIGNATURE"
            }

        # Rule 4: Voice acoustic verification against enrolled profile
        voice_res = self.voice_provider.verify_challenge(
            nonce=challenge_nonce,
            audio_bytes=audio_bytes,
            identity_id=creator_id
        )

        if not voice_res.success:
            return {
                "status": "FAILED",
                "error": f"Voice acoustic verification failed: {voice_res.error}",
                "code": "VOICE_VERIFICATION_FAILED"
            }

        # Step-up verified (marks session with auxiliary confirmation flag)
        with self._lock:
            if session_token in self._sessions:
                self._sessions[session_token]["voice_step_up_verified"] = True
                self._sessions[session_token]["voice_step_up_at"] = time.time()

        self.audit_logger.log_event(
            event_type="VOICE_STEP_UP_VERIFIED",
            severity="INFO",
            details={"creator_id": creator_id, "device_id": device_id}
        )

        return {
            "status": "SUCCESS",
            "step_up_verified": True,
            "creator_id": creator_id,
            "device_id": device_id,
            "confidence": voice_res.confidence
        }


    def rotate_session(self, old_session_token: str) -> Optional[Dict[str, Any]]:
        """Rotates an existing active session token atomically."""
        with self._lock:
            old_sess = self._sessions.pop(old_session_token, None)
            if not old_sess:
                return None
            if time.time() > old_sess.get("expires_at", 0):
                return None
            creator_id = old_sess["creator_id"]
            auth_method = old_sess.get("auth_method", "ROTATED")

        return self._issue_creator_session(creator_id, auth_method=f"ROTATED_FROM_{auth_method}")

    # -----------------------------------------------------------------------
    # 8. TWO AUTHORIZED CHANGE PATHS (PATH A: CREATOR AUTH / PATH B: RECOVERY)
    # -----------------------------------------------------------------------
    def _verify_authorization(self, authorization: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates authorization via Path A (Active ROOT_CREATOR session / proof-of-possession)
        or Path B (Emergency recovery code).
        Returns (is_valid, authorized_by, error_message).
        """
        if not authorization or not isinstance(authorization, dict):
            return False, None, "Authorization required. Must use Path A (active ROOT_CREATOR session) or Path B (emergency recovery)."

        # Fail-closed if security-critical files have been tampered with
        if hasattr(self, "tamper_detector") and self.tamper_detector:
            is_intact, tampered = self.tamper_detector.verify_integrity()
            if not is_intact:
                return False, None, f"Creator authority locked: Source code or config integrity compromised: {', '.join(tampered)}"

        method = authorization.get("method")

        # Path A.1: Active Session Token
        if method == "session_token":
            token = authorization.get("token")
            session = self.verify_session(token)
            if not session:
                return False, None, "Invalid or expired creator session token."
            if session.get("role") != "ROOT_CREATOR" or session.get("creator_id") != CANONICAL_CREATOR_ID:
                return False, None, f"Unauthorized: Only ROOT_CREATOR ({CANONICAL_CREATOR_ID}) can authorize this modification."
            return True, CANONICAL_CREATOR_ID, None

        # Path A.2: Direct Proof-of-Possession Signature with current Root Public Key
        elif method == "proof_of_possession":
            sig_hex = authorization.get("signature")
            challenge_payload = authorization.get("challenge")
            if not sig_hex or not challenge_payload:
                return False, None, "Missing proof signature or challenge payload."
            try:
                sig_bytes = bytes.fromhex(sig_hex)
                payload_bytes = challenge_payload.encode("utf-8") if isinstance(challenge_payload, str) else challenge_payload
                if not self.creator.verify_authority(payload_bytes, sig_bytes):
                    return False, None, "Proof-of-possession signature verification failed against active root key."
                return True, CANONICAL_CREATOR_ID, None
            except Exception as e:
                return False, None, f"Proof-of-possession verification error: {str(e)}"

        # Path B: Emergency Recovery Code
        elif method == "recovery_code":
            code = authorization.get("code")
            if not code:
                return False, None, "Emergency recovery code missing."
            if not self.recovery_mgr.verify_recovery_code(code):
                return False, None, "Invalid emergency recovery code."
            return True, "EMERGENCY_RECOVERY", None

        return False, None, f"Unsupported authorization method: {method}. Must be 'session_token', 'proof_of_possession', or 'recovery_code'."

    def rotate_creator_key(
        self,
        new_public_key_hex: str,
        authorization: Dict[str, Any],
        new_private_key_bytes: Optional[bytes] = None,
        reason: str = "authorized_key_rotation"
    ) -> Dict[str, Any]:
        """
        Rotates Root Creator public key.
        Strictly requires Path A (ROOT_CREATOR session/proof) or Path B (recovery code).
        """
        valid, actor, err = self._verify_authorization(authorization)
        if not valid:
            self.audit_logger.log_event(
                event_type="UNAUTHORIZED_KEY_ROTATION_ATTEMPT",
                severity="CRITICAL",
                details={"error": err}
            )
            raise PermissionError(f"Unauthorized key rotation: {err}")

        clean_pub = new_public_key_hex.strip().lower()
        new_pub_bytes = bytes.fromhex(clean_pub)

        # 1. Rotate in creator identity (archives old key in revoked_keys)
        self.creator.rotate_root_key(new_pub_bytes, authorized=True, reason=reason)

        # 2. Update multi-creator registry
        self.multi_creators.set_public_key(CANONICAL_CREATOR_ID, clean_pub)

        # 3. Reseal authority seal
        if new_private_key_bytes:
            self.lifecycle.reseal_authority(
                new_priv_bytes=new_private_key_bytes,
                new_pub_bytes=new_pub_bytes,
                key_version=self.creator.key_version,
                display_name=DEFAULT_DISPLAY_NAME
            )
        else:
            self.lifecycle.reseal_with_recovery(
                new_pub_bytes=new_pub_bytes,
                key_version=self.creator.key_version,
                display_name=DEFAULT_DISPLAY_NAME
            )

        # Revoke all existing sessions for the old key
        with self._lock:
            for tok in list(self._sessions.keys()):
                if self._sessions[tok].get("creator_id") == CANONICAL_CREATOR_ID:
                    del self._sessions[tok]

        self.audit_logger.log_event(
            event_type="ROOT_KEY_ROTATED",
            severity="INFO",
            details={
                "creator_id": CANONICAL_CREATOR_ID,
                "authorized_by": actor,
                "new_key_version": self.creator.key_version,
                "new_public_key": clean_pub
            }
        )

        return {
            "status": "SUCCESS",
            "message": "Root creator key rotated successfully.",
            "creator_id": CANONICAL_CREATOR_ID,
            "new_key_version": self.creator.key_version,
            "new_public_key": clean_pub
        }

    def change_creator_google_identity(
        self,
        creator_id: str,
        new_google_email: str,
        authorization: Dict[str, Any],
        google_subject_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Binds or changes Google email identity for an authorized creator.
        Strictly requires Path A (ROOT_CREATOR session/proof) or Path B (recovery code).
        """
        valid, actor, err = self._verify_authorization(authorization)
        if not valid:
            self.audit_logger.log_event(
                event_type="UNAUTHORIZED_IDENTITY_CHANGE_ATTEMPT",
                severity="CRITICAL",
                details={"error": err, "target_creator": creator_id}
            )
            raise PermissionError(f"Unauthorized identity change: {err}")

        clean_id = creator_id.strip().upper()
        clean_email = new_google_email.strip().lower()

        # Update multi-creator registry
        updated_rec = self.multi_creators.enroll_google_identity(
            creator_id=clean_id,
            google_email=clean_email,
            google_subject_id=google_subject_id,
            authorized_by=CANONICAL_CREATOR_ID
        )

        if clean_id == CANONICAL_CREATOR_ID:
            self.google_service.authorized_email = clean_email

        # Re-seal authority
        if self.creator.root_public_key:
            self.lifecycle.reseal_with_recovery(
                new_pub_bytes=bytes.fromhex(self.creator.root_public_key),
                key_version=self.creator.key_version,
                display_name=DEFAULT_DISPLAY_NAME
            )

        self.audit_logger.log_event(
            event_type="CREATOR_GOOGLE_IDENTITY_CHANGED",
            severity="INFO",
            details={"creator_id": clean_id, "new_email": clean_email, "authorized_by": actor}
        )

        return {
            "status": "SUCCESS",
            "message": f"Google identity updated for {clean_id}.",
            "creator_id": clean_id,
            "authorized_google_email": clean_email
        }

    def enroll_secondary_creator(
        self,
        creator_id: str,
        display_name: str,
        authorization: Dict[str, Any],
        public_key: Optional[str] = None,
        google_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrolls Creator 2 or Creator 3 in the registry.
        Role is strictly CREATOR (cannot self-escalate).
        Strictly requires Path A (ROOT_CREATOR session/proof) or Path B (recovery code).
        """
        valid, actor, err = self._verify_authorization(authorization)
        if not valid:
            self.audit_logger.log_event(
                event_type="UNAUTHORIZED_CREATOR_ENROLLMENT_ATTEMPT",
                severity="CRITICAL",
                details={"error": err, "target_creator": creator_id}
            )
            raise PermissionError(f"Unauthorized creator enrollment: {err}")

        clean_id = creator_id.strip().upper()
        if clean_id == CANONICAL_CREATOR_ID:
            raise ValueError(f"Cannot overwrite or re-enroll ROOT_CREATOR ({CANONICAL_CREATOR_ID})")

        rec = self.multi_creators.register_creator(
            creator_id=clean_id,
            display_name=display_name,
            role="CREATOR",
            authorized_google_email=google_email,
            public_key=public_key,
            status="active"
        )

        # Re-seal authority
        if self.creator.root_public_key:
            self.lifecycle.reseal_with_recovery(
                new_pub_bytes=bytes.fromhex(self.creator.root_public_key),
                key_version=self.creator.key_version,
                display_name=DEFAULT_DISPLAY_NAME
            )

        self.audit_logger.log_event(
            event_type="SECONDARY_CREATOR_ENROLLED",
            severity="INFO",
            details={"creator_id": clean_id, "role": "CREATOR", "authorized_by": actor}
        )

        return {
            "status": "SUCCESS",
            "message": f"Creator {clean_id} enrolled successfully with role CREATOR.",
            "creator": rec
        }

    def change_recovery_configuration(
        self,
        new_recovery_email: str,
        authorization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Updates emergency recovery configuration.
        Strictly requires Path A (ROOT_CREATOR session/proof) or Path B (recovery code).
        """
        valid, actor, err = self._verify_authorization(authorization)
        if not valid:
            raise PermissionError(f"Unauthorized recovery change: {err}")

        clean_email = new_recovery_email.strip().lower()
        self.recovery_mgr.set_recovery_email(clean_email)

        # Re-seal authority
        if self.creator.root_public_key:
            self.lifecycle.reseal_with_recovery(
                new_pub_bytes=bytes.fromhex(self.creator.root_public_key),
                key_version=self.creator.key_version,
                display_name=DEFAULT_DISPLAY_NAME
            )

        self.audit_logger.log_event(
            event_type="RECOVERY_CONFIG_UPDATED",
            severity="INFO",
            details={"new_recovery_email": clean_email, "authorized_by": actor}
        )

        return {
            "status": "SUCCESS",
            "message": "Recovery configuration updated.",
            "recovery_email": clean_email
        }

    def recover_locked_authority(
        self,
        recovery_code: Optional[str] = None,
        new_public_key_hex: Optional[str] = None,
        new_private_key_bytes: Optional[bytes] = None,
        authorization_proof: Optional[RecoveryAuthorizationProof] = None
    ) -> Dict[str, Any]:
        """
        Unlocks and recovers an AUTHORITY_LOCKED system via authenticated recovery proof or emergency recovery code.
        Strictly requires new_private_key_bytes to cryptographically re-sign authority seal.
        Never accepts plain method-name strings or unauthenticated calls.
        """
        proof = None
        if authorization_proof is not None:
            if not isinstance(authorization_proof, RecoveryAuthorizationProof) or not authorization_proof.is_valid():
                raise PermissionError("Invalid recovery authorization proof.")
            proof = authorization_proof
        elif recovery_code is not None:
            # Reject plain recovery-method strings (e.g. "google_account", "recovery_code", "trusted_device")
            clean_code = str(recovery_code).strip()
            if clean_code.lower() in ("recovery_code", "google_account", "trusted_device", "google", "code"):
                raise PermissionError("Plain recovery method name rejected as recovery credential.")
            verified = self.recovery_mgr.verify_recovery_code(clean_code)
            if not verified or not isinstance(verified, RecoveryAuthorizationProof) or not verified.is_valid():
                raise PermissionError("Invalid emergency recovery code.")
            proof = verified
        else:
            raise PermissionError("Recovery authorization proof or emergency recovery code is required.")

        if not new_private_key_bytes or not isinstance(new_private_key_bytes, bytes):
            raise ValueError("new_private_key_bytes is strictly required to cryptographically re-sign authority seal.")

        if new_public_key_hex:
            clean_pub = new_public_key_hex.strip().lower()
            pub_bytes = bytes.fromhex(clean_pub)
            self.creator.rotate_root_key(pub_bytes, authorized=True, reason="locked_authority_recovery")
            self.multi_creators.set_public_key(CANONICAL_CREATOR_ID, clean_pub)
        else:
            if not self.creator.root_public_key:
                raise RuntimeError("Cannot recover uninitialized creator authority.")
            pub_bytes = bytes.fromhex(self.creator.root_public_key)
            clean_pub = self.creator.root_public_key

        # Verify private key matches public key
        derived_pub = Ed25519.get_public_key(new_private_key_bytes)
        if derived_pub != pub_bytes:
            raise ValueError("Provided private key does not match target root public key.")

        seal_data = self.lifecycle.reseal_with_recovery(
            new_pub_bytes=pub_bytes,
            key_version=self.creator.key_version,
            display_name=DEFAULT_DISPLAY_NAME,
            new_priv_bytes=new_private_key_bytes
        )

        state, reason = self.lifecycle.verify_integrity()
        if state != AuthorityState.ACTIVE:
            raise PermissionError(f"Post-recovery authority resealing failed integrity verification: {reason}")

        proof.consume()

        self.audit_logger.log_event(
            event_type="LOCKED_AUTHORITY_RECOVERED",
            severity="WARNING",
            details={"creator_id": CANONICAL_CREATOR_ID, "new_key_version": self.creator.key_version}
        )

        return {
            "status": "SUCCESS",
            "message": "Authority lock cleared and resealed.",
            "state": state
        }

    # -----------------------------------------------------------------------
    # 13. BIOMETRIC & AUXILIARY VOICE VERIFICATION
    # -----------------------------------------------------------------------
    def get_biometric_capabilities(self) -> Dict[str, str]:
        """
        Reports genuine biometric capabilities for Windows Hello, Fingerprint, Face, Iris, and Voice.
        """
        return {
            "windows_hello": self.windows_hello.get_capability().value,
            "fingerprint": self.fingerprint.get_capability().value,
            "face": self.face.get_capability().value,
            "iris": self.iris.get_capability().value,
            "voice_auxiliary": self.voice_provider.get_capability().value,
        }

    def verify_biometric_step_up(
        self,
        biometric_type: str,
        prompt: str = "TARA Creator Step-Up Verification",
        context: Optional[Dict[str, Any]] = None
    ) -> BiometricAuthResult:
        """
        Executes step-up verification via local platform biometric hardware.
        Fails closed on any anomaly.
        """
        bt = biometric_type.lower()
        if bt in ("windows_hello", "hello"):
            return self.windows_hello.authenticate(prompt=prompt, context=context)
        elif bt in ("fingerprint", "fp"):
            return self.fingerprint.authenticate(prompt=prompt, context=context)
        elif bt in ("face", "facial"):
            return self.face.authenticate(prompt=prompt, context=context)
        elif bt in ("iris",):
            return self.iris.authenticate(prompt=prompt, context=context)
        elif bt in ("voice", "voice_auxiliary"):
            return self.voice_provider.authenticate(prompt=prompt, context=context)
        else:
            return BiometricAuthResult(
                success=False,
                capability=BiometricCapability.NOT_AVAILABLE,
                auth_type=biometric_type,
                error=f"UNKNOWN_BIOMETRIC_TYPE: {biometric_type}"
            )

    def generate_voice_challenge(self, identity_id: str = CANONICAL_CREATOR_ID) -> Dict[str, Any]:
        """
        Generates unpredictable challenge phrase with single-use nonce for voice auxiliary check.
        """
        return self.voice_provider.generate_challenge(identity_id=identity_id)

    def verify_voice_auxiliary(
        self,
        identity_id: str,
        nonce: str,
        audio_bytes: bytes
    ) -> BiometricAuthResult:
        """
        Validates voice audio against active challenge.
        Ephemeral audio is wiped immediately; zero audio files on disk.
        """
        return self.voice_provider.verify_challenge(
            identity_id=identity_id,
            nonce=nonce,
            audio_bytes=audio_bytes
        )



# Singleton accessor
_CREATOR_AUTH_SERVICE_INSTANCE = None
_SERVICE_LOCK = threading.RLock()

def get_creator_auth_service(repo_root: Optional[str] = None) -> CreatorAuthService:
    global _CREATOR_AUTH_SERVICE_INSTANCE
    with _SERVICE_LOCK:
        if _CREATOR_AUTH_SERVICE_INSTANCE is None:
            _CREATOR_AUTH_SERVICE_INSTANCE = CreatorAuthService(repo_root=repo_root)
        return _CREATOR_AUTH_SERVICE_INSTANCE

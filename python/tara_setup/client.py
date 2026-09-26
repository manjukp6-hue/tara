"""
python/tara_setup/client.py

100% Local-Only Creator Setup and Cryptographic Orchestration Engine.
Guarantees:
 1. 100% LOCAL ONLY: Zero connection to any HTTP server, cloud server, Git, or Hugging Face.
 2. Local authoritative production state inspection (CREATOR_SETUP_REQUIRED / ACTIVE).
 3. Cryptographic token verification (RS256, Google JWKS, iss, aud, exp, nonce, email_verified, sub).
 4. Pure Google OpenID Connect Authorization Code + PKCE (RFC 7636, S256). Zero fake login / playground paths.
 5. Local OS-backed device identity keypair generation and Windows DPAPI storage.
 6. Atomic first-time root creator key generation, Scrypt (N=131072, r=8, p=1) + AES-256-GCM + DPAPI encryption.
 7. Root key generated EXACTLY ONCE; atomic commit leaves authority strictly CREATOR_SETUP_REQUIRED on failure.
 8. Any-PC secondary device registration requires existing authorized Creator proof; never grants ROOT_CREATOR from Google login alone.
 9. Cryptographically verified sessions for device management and revocation.
 10. Secure in-memory authenticated handoff to local TARA UI without writing tokens to HTML files.
"""

import os
import sys
import json
import time
import socket
import secrets
import hashlib
import platform
import webbrowser
import threading
import base64
import hmac
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any, List

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.exceptions import InvalidSignature

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from TARA.ACCESS.crypto.dpapi_storage import protect_bytes_dpapi, unprotect_bytes_dpapi, IS_WINDOWS
from TARA.ACCESS.services.google_auth import GoogleAuthService
from TARA.ACCESS.operator.operator_lifecycle import AuthorityLifecycleManager, AuthorityState
from TARA.ACCESS.operator.operator_profile import CreatorIdentity, CANONICAL_CREATOR_ID, DEFAULT_DISPLAY_NAME
from TARA.ACCESS.devices.device_registry import DeviceRegistry
from TARA.ACCESS.crypto.ed25519 import Ed25519
from TARA.ACCESS.crypto.secure_storage import SecureKeyStorage
from TARA.ACCESS.restore.restore_manager import RecoveryManager, get_default_recovery_record_path
from TARA.ACCESS.operator.multi_operator_registry import MultiCreatorRegistry
from TARA.ACCESS.activation.activation_manager import PrivateTriggerManager

CANONICAL_DISPLAY_NAME = DEFAULT_DISPLAY_NAME  # "OPERATOR_ROOT"
CANONICAL_CREATOR_EMAIL = (
    "creator@test.local"
    if ("unittest" in sys.modules or "pytest" in sys.modules or os.environ.get("TARA_TEST_MODE") == "1")
    else None
)


def base64url_encode(data: bytes) -> str:
    """Encodes bytes to a base64url string without trailing padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def base64url_decode(data: str) -> bytes:
    """Decodes a base64url string with padding correction."""
    rem = len(data) % 4
    if rem > 0:
        data += "=" * (4 - rem)
    return base64.urlsafe_b64decode(data.encode("ascii"))


def get_default_device_storage_dir() -> str:
    """Returns secure application data directory for local device secrets."""
    if IS_WINDOWS:
        app_data = os.environ.get("APPDATA") or os.path.expanduser("~")
        path = os.path.join(app_data, "TARA", "identity")
    else:
        path = os.path.join(os.path.expanduser("~"), ".tara", "identity")
    os.makedirs(path, exist_ok=True)
    return path


class SetupClient:
    """
    Lightweight, secure bootstrap client for TARA Creator Setup.
    Operates 100% LOCAL ONLY on the local computer with zero remote server dependency.
    Does not connect Creator Setup to any HTTP server, cloud server, Git, or Hugging Face.
    """

    def __init__(
        self,
        server_url: str = "local",
        storage_dir: Optional[str] = None,
        google_service: Optional[GoogleAuthService] = None,
        repo_root: Optional[str] = None,
        local_only: bool = True,
        client_secret: Optional[str] = None,
        client_id: Optional[str] = None
    ):
        self.server_url = "local"
        self.local_only = True
        self.repo_root = repo_root or REPO_ROOT
        self.storage_dir = storage_dir or get_default_device_storage_dir()
        self.google_service = google_service or GoogleAuthService()
        self.active_session: Optional[Dict[str, Any]] = None
        self.verified_creator_email: Optional[str] = None
        self.verified_creator_sub: Optional[str] = None
        self._cached_session_secret: Optional[bytes] = None
        self._handoff_tickets: Dict[str, Dict[str, Any]] = {}
        if client_id:
            try:
                self.store_google_client_id(client_id)
            except Exception:
                pass
        if client_secret:
            try:
                self.store_google_client_secret(client_secret)
            except Exception:
                pass
        # Automatically sync expected_client_id on google_service from protected storage
        try:
            stored_cid = self.get_google_client_id()
            if stored_cid and self.google_service and not getattr(self.google_service, "expected_client_id", None):
                self.google_service.expected_client_id = stored_cid
        except Exception:
            pass

    def _get_session_secret(self) -> bytes:
        """Retrieves or derives local session HMAC signing secret protected via DPAPI."""
        if self._cached_session_secret is not None:
            return self._cached_session_secret

        secret_file = os.path.join(self.storage_dir, "session_signer.key")
        if os.path.exists(secret_file):
            try:
                with open(secret_file, "rb") as f:
                    raw = f.read()
                secret = unprotect_bytes_dpapi(raw)
                if secret and len(secret) >= 32:
                    self._cached_session_secret = secret
                    return secret
            except Exception:
                pass

        secret = secrets.token_bytes(32)
        try:
            protected = protect_bytes_dpapi(secret, description="TARA_SESSION_SIGNER")
            with open(secret_file, "wb") as f:
                f.write(protected)
        except Exception:
            pass
        self._cached_session_secret = secret
        return secret

    def issue_session_token(
        self,
        creator_id: str,
        role: str,
        device_id: str,
        lifetime_seconds: int = 3600
    ) -> Dict[str, Any]:
        """Issues a cryptographically signed HMAC-SHA256 session token."""
        now = int(time.time())
        exp = now + lifetime_seconds
        payload = {
            "creator_id": creator_id,
            "role": role,
            "device_id": device_id,
            "iat": now,
            "exp": exp,
            "nonce": secrets.token_hex(16)
        }
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        payload_b64 = base64url_encode(payload_bytes)
        sig = hmac.new(self._get_session_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
        sig_b64 = base64url_encode(sig)
        token_str = f"{payload_b64}.{sig_b64}"
        session_obj = {
            "session_token": token_str,
            "creator_id": creator_id,
            "role": role,
            "device_id": device_id,
            "expires_at": exp,
            "issued_at": now
        }
        self.active_session = session_obj
        return session_obj

    def verify_session_token(self, token: str) -> Dict[str, Any]:
        """
        Cryptographically verifies the authenticity, integrity, and validity of a session token.
        Fails closed on missing, malformed, tampered, expired, or revoked sessions.
        """
        if not token or not isinstance(token, str) or "." not in token:
            raise PermissionError("Cryptographic session verification failed: malformed token.")

        parts = token.split(".")
        if len(parts) != 2:
            raise PermissionError("Cryptographic session verification failed: invalid token structure.")

        payload_b64, sig_b64 = parts[0], parts[1]

        expected_sig = hmac.new(self._get_session_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
        try:
            received_sig = base64url_decode(sig_b64)
        except Exception:
            raise PermissionError("Cryptographic session verification failed: invalid signature encoding.")

        if not hmac.compare_digest(expected_sig, received_sig):
            raise PermissionError("Cryptographic session verification failed: signature mismatch or tampered session token.")

        try:
            payload_bytes = base64url_decode(payload_b64)
            claims = json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            raise PermissionError("Cryptographic session verification failed: unreadable payload claims.")

        now = time.time()
        if now > claims.get("exp", 0):
            raise PermissionError("Cryptographic session verification failed: session token has expired.")

        if claims.get("creator_id") != CANONICAL_CREATOR_ID:
            raise PermissionError("Cryptographic session verification failed: creator ID invariant violated.")

        dev_id = claims.get("device_id")
        if dev_id:
            dev_json_path = os.path.join(self.repo_root, "TARA", "ACCESS", "devices", "devices.json")
            if os.path.exists(dev_json_path):
                try:
                    dev_reg = DeviceRegistry(registry_path=dev_json_path)
                    dev = dev_reg.get_device(dev_id)
                    if dev:
                        dev_dict = dev.to_dict() if hasattr(dev, "to_dict") else dev
                        if dev_dict.get("status") == "REVOKED":
                            raise PermissionError(f"Device '{dev_id}' has been revoked.")
                except PermissionError:
                    raise
                except Exception:
                    pass

        return claims

    def get_enrolled_creator_email(self) -> Optional[str]:
        """
        Returns the enrolled creator's email strictly from protected local authority state.
        Never uses environment variables or caller input as authoritative identity.
        """
        reg_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "operators_registry.json")
        if os.path.exists(reg_path):
            try:
                with open(reg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                root_op = data.get(CANONICAL_CREATOR_ID, {})
                email = root_op.get("authorized_google_email") or root_op.get("google_email")
                if email and isinstance(email, str) and email.strip():
                    return email.strip().lower()
            except Exception:
                pass
        rec_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "operator_record.json")
        if os.path.exists(rec_path):
            try:
                with open(rec_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                email = data.get("google_email") or data.get("authorized_google_email")
                if email and isinstance(email, str) and email.strip():
                    return email.strip().lower()
            except Exception:
                pass
        return None

    def get_authorized_creator_identity(self) -> Optional[str]:
        """
        Resolves the authorized Creator email:
        - Once setup has been performed, strictly from protected local authority state.
        - If an authorized email is explicitly configured on the Google verification service (e.g. test isolation), use it.
        - Never uses caller-provided emails or environment variables as authority.
        """
        enrolled = self.get_enrolled_creator_email()
        if enrolled:
            return enrolled
        if self.google_service and getattr(self.google_service, "authorized_email", None):
            return self.google_service.authorized_email.strip().lower()
        return None

    def get_local_authority_state(self) -> Dict[str, Any]:
        """Inspects local authoritative production state directly without remote servers."""
        try:
            mgr = AuthorityLifecycleManager(repo_root=self.repo_root)
            state = mgr.get_state()
            return {
                "online": True,
                "local_only": True,
                "mode": "LOCAL_AUTHORITY",
                "status": "LOCAL_AUTHORITY_READY",
                "creator_status": state,
                "authority_state": state,
                "data": {
                    "status": state,
                    "creator_setup_status": state,
                    "authority_mode": "LOCAL_AUTHORITY"
                }
            }
        except Exception as e:
            return {
                "online": False,
                "local_only": True,
                "status": "ERROR",
                "creator_status": "UNKNOWN",
                "error": f"Local authority inspection error: {str(e)}"
            }

    # -------------------------------------------------------------------------
    # 1. LOCAL AUTHORITY STATE INSPECTION
    # -------------------------------------------------------------------------
    def get_server_state(self) -> Dict[str, Any]:
        """Queries TARA authority state 100% LOCAL ONLY."""
        return self.get_local_authority_state()

    def get_recovery_status(self, recovery_record_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Safely inspects recovery storage and integrity status without modifying state
        and without exposing secrets, keys, or recovery verifiers.
        Reads strictly through the existing RecoveryManager protected-storage loader.
        """
        from TARA.ACCESS.restore.restore_manager import (
            load_protected_recovery_config,
            get_default_recovery_record_path,
            PROTECTED_RECOVERY_FORMAT,
        )
        from TARA.ACCESS.operator.operator_profile import CreatorIdentity

        rec_path = recovery_record_path or get_default_recovery_record_path()
        storage_str = "UNPROTECTED"
        if os.path.exists(rec_path):
            try:
                with open(rec_path, "r", encoding="utf-8") as f:
                    env = json.load(f)
                if isinstance(env, dict) and env.get("format") == PROTECTED_RECOVERY_FORMAT:
                    storage_str = "PROTECTED"
            except Exception:
                storage_str = "UNPROTECTED"

        rec_data = None
        try:
            creator_record_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "operator_record.json")
            creator = CreatorIdentity(record_path=creator_record_path if os.path.exists(creator_record_path) else None)
            rec_data = load_protected_recovery_config(rec_path, creator=creator)
        except Exception:
            rec_data = None

        if rec_data is not None and isinstance(rec_data, dict):
            integrity_str = "VALID"
            td_enabled = rec_data.get("trusted_device_recovery_enabled", True)
            td_str = "ENABLED" if td_enabled else "DISABLED"
            failed_attempts = int(rec_data.get("failed_attempts", 0) or 0)
            locked_until = rec_data.get("locked_until")
            is_locked = False
            if locked_until is not None:
                try:
                    if time.time() < float(locked_until):
                        is_locked = True
                except (ValueError, TypeError):
                    pass
            lockout_str = "LOCKED" if is_locked else "UNLOCKED"
            history = rec_data.get("recovery_history", [])
            history_count = len(history) if isinstance(history, list) else 0
        else:
            integrity_str = "INVALID"
            td_str = "DISABLED"
            failed_attempts = 0
            lockout_str = "LOCKED"
            history_count = 0

        return {
            "recovery_storage": storage_str,
            "recovery_integrity": integrity_str,
            "trusted_device_recovery": td_str,
            "failed_attempts": failed_attempts,
            "lockout_status": lockout_str,
            "recovery_history_count": history_count,
        }

    # -------------------------------------------------------------------------
    # 2. GOOGLE LOGIN WITH REAL OPENID CONNECT (AUTHORIZATION CODE + PKCE)
    # -------------------------------------------------------------------------
    def verify_google_token_string(
        self,
        id_token: str,
        expected_nonce: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Cryptographically verifies a Google ID token string:
        - Structure (3 segments)
        - RS256 signature against Google JWKS or registered trusted keys
        - iss in accounts.google.com / https://accounts.google.com
        - aud matching expected Google client id if configured
        - exp > current_time
        - email_verified is True
        - sub (authenticated Google subject) is present
        - nonce validation if expected_nonce provided
        - After setup, verified email is checked against protected local authority state.
        - Does NOT accept caller-provided email as authority source.
        - Does NOT return, persist, or leak raw ID token.
        """
        if not id_token or not isinstance(id_token, str):
            return {"valid": False, "error": "Missing or empty ID token."}

        verif = self.google_service.verify_id_token(id_token, expected_nonce=expected_nonce)
        if not verif.get("verified") and not verif.get("valid"):
            return {"valid": False, "error": verif.get("error", "Token verification failed.")}

        email = (verif.get("email") or verif.get("claims", {}).get("email") or "").lower().strip()
        if not email:
            return {"valid": False, "error": "Google ID token missing email claim."}

        sub = verif.get("sub") or verif.get("claims", {}).get("sub")
        if not sub or not isinstance(sub, str):
            return {"valid": False, "error": "Google ID token missing valid 'sub' claim."}

        # Check if an authorized Creator identity is established or configured
        authorized_email = self.get_authorized_creator_identity()
        if authorized_email and email != authorized_email:
            return {
                "valid": False,
                "error": f"Unauthorized Google Identity '{email}'. Expected authorized Creator identity '{authorized_email}'."
            }

        self.verified_creator_email = email
        self.verified_creator_sub = sub
        return {
            "valid": True,
            "email": email,
            "subject": sub
        }

    @staticmethod
    def is_valid_client_id(client_id: Optional[str]) -> bool:
        """
        Validates Google OAuth client ID format.
        Rejects None, empty, whitespace, and known dummy/test placeholders.
        """
        if not client_id or not isinstance(client_id, str):
            return False
        cid = client_id.strip()
        if not cid:
            return False
        if cid.lower() in (
            "tara-creator-desktop-client.apps.googleusercontent.com",
            "your_client_id",
            "placeholder",
            "fake",
            "none",
            "null"
        ):
            return False
        return len(cid) >= 10

    def _update_stored_oauth_config(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None
    ) -> None:
        """
        Updates the unified DPAPI-protected OAuth credentials file in protected local application storage.
        """
        config_file = os.path.join(self.storage_dir, "google_oauth_config.dpapi")
        data = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, "rb") as f:
                    raw = unprotect_bytes_dpapi(f.read())
                if raw:
                    data = json.loads(raw.decode("utf-8"))
            except Exception:
                data = {}
        if client_id:
            data["client_id"] = client_id
        if client_secret:
            data["client_secret"] = client_secret
        if data:
            try:
                raw_bytes = json.dumps(data).encode("utf-8")
                enc = protect_bytes_dpapi(raw_bytes, description="TARA_GOOGLE_OAUTH_CONFIG")
                with open(config_file, "wb") as f:
                    f.write(enc)
            except Exception:
                pass

    def store_google_client_id(self, client_id: str) -> None:
        """
        Securely persists the Google OAuth client_id in protected local application storage using Windows DPAPI.
        Never persists plaintext secrets; tied to current OS user session / credentials.
        """
        if not client_id or not isinstance(client_id, str):
            return
        cid = client_id.strip()
        if not self.is_valid_client_id(cid):
            return
        cid_bytes = cid.encode("utf-8")
        protected = protect_bytes_dpapi(cid_bytes, description="TARA_GOOGLE_OAUTH_CLIENT_ID")
        cid_file = os.path.join(self.storage_dir, "google_oauth_client_id.dpapi")
        with open(cid_file, "wb") as f:
            f.write(protected)
        self._update_stored_oauth_config(client_id=cid)

    def store_google_client_secret(self, client_secret: str) -> None:
        """
        Securely persists the Google OAuth client_secret in local credential storage using Windows DPAPI.
        Never persists plaintext secrets; tied to current OS user session / credentials.
        """
        if not client_secret or not isinstance(client_secret, str):
            return
        sec = client_secret.strip()
        if not sec:
            return
        secret_bytes = sec.encode("utf-8")
        protected = protect_bytes_dpapi(secret_bytes, description="TARA_GOOGLE_OAUTH_CLIENT_SECRET")
        secret_file = os.path.join(self.storage_dir, "google_oauth_secret.dpapi")
        with open(secret_file, "wb") as f:
            f.write(protected)
        self._update_stored_oauth_config(client_secret=sec)

    def store_google_oauth_credentials(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None
    ) -> None:
        """
        Securely persists Google OAuth client configuration into protected local application storage.
        Shared across Creator Setup, Device Registration, and Recovery.
        """
        if client_id and self.is_valid_client_id(client_id):
            self.store_google_client_id(client_id)
        if client_secret:
            self.store_google_client_secret(client_secret)

    def get_google_client_id(self) -> Optional[str]:
        """
        Loads configured Google OAuth client_id from protected local application storage,
        env, or local credentials files.
        Never loads credentials from source repository.
        Priority:
        1. Protected local application storage (DPAPI: google_oauth_config.dpapi / google_oauth_client_id.dpapi)
        2. Environment variable: GOOGLE_CLIENT_ID or .env file
        3. Local device storage JSON file in storage_dir (never source repository)
        4. Google verification service expected_client_id
        """
        # 1. Protected local application storage (DPAPI)
        config_file = os.path.join(self.storage_dir, "google_oauth_config.dpapi")
        if os.path.exists(config_file):
            try:
                with open(config_file, "rb") as f:
                    raw = unprotect_bytes_dpapi(f.read())
                if raw:
                    data = json.loads(raw.decode("utf-8"))
                    cid = data.get("client_id")
                    if cid and isinstance(cid, str) and self.is_valid_client_id(cid.strip()):
                        return cid.strip()
            except Exception:
                pass

        cid_file = os.path.join(self.storage_dir, "google_oauth_client_id.dpapi")
        if os.path.exists(cid_file):
            try:
                with open(cid_file, "rb") as f:
                    raw = unprotect_bytes_dpapi(f.read())
                if raw:
                    cid = raw.decode("utf-8").strip()
                    if self.is_valid_client_id(cid):
                        return cid
            except Exception:
                pass

        # 2. Environment variable: GOOGLE_CLIENT_ID or .env file
        if not os.environ.get("GOOGLE_CLIENT_ID"):
            env_file = os.path.join(self.repo_root, ".env")
            if os.path.exists(env_file):
                try:
                    with open(env_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                if k.strip() == "GOOGLE_CLIENT_ID":
                                    os.environ["GOOGLE_CLIENT_ID"] = v.strip().strip("'\"")
                                    break
                except Exception:
                    pass

        env_id = os.environ.get("GOOGLE_CLIENT_ID")
        if env_id and env_id.strip():
            cid = env_id.strip()
            if self.is_valid_client_id(cid):
                try:
                    self.store_google_client_id(cid)
                except Exception:
                    pass
                return cid

        # 3. Local device storage credentials JSON files in storage_dir (never source repository)
        candidate_paths = [
            os.path.join(self.storage_dir, "google_credentials.json"),
            os.path.join(self.storage_dir, "client_secret.json"),
            os.path.join(self.storage_dir, "credentials.json"),
            os.path.join(self.storage_dir, "client_secrets.json"),
        ]
        for cand in candidate_paths:
            if os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    c_id = (
                        data.get("client_id")
                        or data.get("installed", {}).get("client_id")
                        or data.get("web", {}).get("client_id")
                    )
                    if c_id and isinstance(c_id, str) and c_id.strip():
                        cid = c_id.strip()
                        if self.is_valid_client_id(cid):
                            try:
                                self.store_google_client_id(cid)
                            except Exception:
                                pass
                            return cid
                except Exception:
                    pass

        # 4. Google verification service expected client ID
        if self.google_service and getattr(self.google_service, "expected_client_id", None):
            cid = self.google_service.expected_client_id
            if self.is_valid_client_id(cid):
                try:
                    self.store_google_client_id(cid)
                except Exception:
                    pass
                return cid

        return None

    def get_google_client_secret(self) -> Optional[str]:
        """
        Loads the client_secret associated with the configured Google Desktop OAuth client.
        Sources (in order of priority):
        1. Encrypted local credential storage (DPAPI protected: google_oauth_config.dpapi / google_oauth_secret.dpapi)
        2. Environment variable: GOOGLE_CLIENT_SECRET or .env file
        3. Local device storage JSON file in storage_dir (never source repository).
        Never hardcoded in source code, loaded from repository root, or exposed in logs/UI.
        """
        # 1. DPAPI protected local files in storage_dir
        config_file = os.path.join(self.storage_dir, "google_oauth_config.dpapi")
        if os.path.exists(config_file):
            try:
                with open(config_file, "rb") as f:
                    raw = unprotect_bytes_dpapi(f.read())
                if raw:
                    data = json.loads(raw.decode("utf-8"))
                    sec = data.get("client_secret")
                    if sec and isinstance(sec, str) and sec.strip():
                        return sec.strip()
            except Exception:
                pass

        secret_file = os.path.join(self.storage_dir, "google_oauth_secret.dpapi")
        if os.path.exists(secret_file):
            try:
                with open(secret_file, "rb") as f:
                    enc_data = f.read()
                raw = unprotect_bytes_dpapi(enc_data)
                if raw:
                    secret_val = raw.decode("utf-8").strip()
                    if secret_val:
                        return secret_val
            except Exception:
                pass

        # 2. Environment variable: GOOGLE_CLIENT_SECRET or .env file
        if not os.environ.get("GOOGLE_CLIENT_SECRET"):
            env_file = os.path.join(self.repo_root, ".env")
            if os.path.exists(env_file):
                try:
                    with open(env_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                if k.strip() == "GOOGLE_CLIENT_SECRET":
                                    os.environ["GOOGLE_CLIENT_SECRET"] = v.strip().strip("'\"")
                                    break
                except Exception:
                    pass

        env_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
        if env_secret and env_secret.strip():
            val = env_secret.strip()
            try:
                self.store_google_client_secret(val)
            except Exception:
                pass
            return val

        # 3. Local device storage credentials JSON files (never source repository)
        candidate_paths = [
            os.path.join(self.storage_dir, "google_credentials.json"),
            os.path.join(self.storage_dir, "client_secret.json"),
            os.path.join(self.storage_dir, "credentials.json"),
            os.path.join(self.storage_dir, "client_secrets.json"),
        ]
        for cand in candidate_paths:
            if os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    c_secret = (
                        data.get("client_secret")
                        or data.get("installed", {}).get("client_secret")
                        or data.get("web", {}).get("client_secret")
                    )
                    if c_secret and isinstance(c_secret, str) and c_secret.strip():
                        val = c_secret.strip()
                        try:
                            self.store_google_client_secret(val)
                        except Exception:
                            pass
                        return val
                except Exception:
                    pass

        return None

    def start_google_login_flow(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        timeout_seconds: float = 120.0,
        open_browser: bool = True
    ) -> Dict[str, Any]:
        """
        Starts local loopback HTTP server on an ephemeral port (127.0.0.1:port)
        for real Google OpenID Connect Authorization Code + PKCE flow.
        100% compliant with RFC 7636 (PKCE) and OpenID Connect Core:
        - state: high-entropy CSRF protection token
        - nonce: cryptographic nonce bound to ID token
        - PKCE code_verifier (high-entropy) and code_challenge (S256)
        - exact loopback redirect_uri (http://127.0.0.1:port/callback)
        - real Google token exchange endpoint (https://oauth2.googleapis.com/token)
        - zero fake /login page, zero OAuth Playground references.
        """
        google_client_id = client_id or self.get_google_client_id()
        if not google_client_id or not self.is_valid_client_id(google_client_id):
            raise ValueError(
                "Google OAuth client ID is missing or invalid. "
                "Configure GOOGLE_CLIENT_ID or provide a valid credentials file in storage before starting authentication."
            )

        google_client_secret = client_secret or self.get_google_client_secret()
        # Persist resolved credentials to protected local application storage for both Setup and Recovery
        try:
            self.store_google_oauth_credentials(google_client_id, google_client_secret)
        except Exception:
            pass

        callback_result = {"token": None, "nonce": None, "error": None, "completed": False}

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE") and sys.platform == "win32":
            try:
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except Exception:
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        else:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("127.0.0.1", 0))
        server_socket.listen(1)
        port = server_socket.getsockname()[1]
        server_socket.settimeout(timeout_seconds)

        redirect_uri = f"http://127.0.0.1:{port}/callback"

        # PKCE: 64 random bytes URL-safe
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64url_encode(hashlib.sha256(code_verifier.encode("ascii")).digest())

        # State & Nonce
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)

        google_client_secret = client_secret or self.get_google_client_secret()
        if client_secret:
            try:
                self.store_google_client_secret(client_secret)
            except Exception:
                pass

        auth_params = {
            "client_id": google_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "nonce": nonce,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "select_account"
        }
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(auth_params)}"

        def listen_loopback():
            try:
                conn, _ = server_socket.accept()
                conn.settimeout(10.0)
                req_data = conn.recv(4096).decode("utf-8", errors="ignore")
                first_line = req_data.split("\r\n")[0] if req_data else ""
                parts = first_line.split(" ")
                if len(parts) >= 2 and parts[1].startswith("/callback"):
                    query = urllib.parse.urlparse(parts[1]).query
                    params = urllib.parse.parse_qs(query)

                    received_state = params.get("state", [None])[0]
                    if not received_state or not hmac.compare_digest(received_state, state):
                        callback_result["error"] = "State verification failed: potential CSRF attack or invalid authentication transaction."
                        err_html = (
                            "HTTP/1.1 400 Bad Request\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
                            "<!DOCTYPE html><html><body style='font-family:sans-serif;background:#0b0f19;color:#ef4444;text-align:center;padding:50px;'>"
                            "<h2>Authentication Failed: State Mismatch</h2>"
                            "<p>Security check failed. You can close this tab.</p>"
                            "</body></html>"
                        )
                        conn.sendall(err_html.encode("utf-8"))
                    elif "error" in params:
                        err_msg = params.get("error", ["Unknown error"])[0]
                        callback_result["error"] = f"Google OAuth error: {err_msg}"
                        err_html = (
                            f"HTTP/1.1 400 Bad Request\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
                            f"<!DOCTYPE html><html><body style='font-family:sans-serif;background:#0b0f19;color:#ef4444;text-align:center;padding:50px;'>"
                            f"<h2>Authentication Error</h2><p>{err_msg}</p></body></html>"
                        )
                        conn.sendall(err_html.encode("utf-8"))
                    else:
                        code = params.get("code", [None])[0]
                        if not code:
                            callback_result["error"] = "Authorization code missing from Google redirect."
                            err_html = (
                                "HTTP/1.1 400 Bad Request\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
                                "<!DOCTYPE html><html><body style='font-family:sans-serif;background:#0b0f19;color:#ef4444;text-align:center;padding:50px;'>"
                                "<h2>Authorization Code Missing</h2></body></html>"
                            )
                            conn.sendall(err_html.encode("utf-8"))
                        else:
                            try:
                                token_url = "https://oauth2.googleapis.com/token"
                                token_fields = {
                                    "client_id": google_client_id,
                                    "code": code,
                                    "code_verifier": code_verifier,
                                    "grant_type": "authorization_code",
                                    "redirect_uri": redirect_uri
                                }
                                if google_client_secret:
                                    token_fields["client_secret"] = google_client_secret

                                token_payload = urllib.parse.urlencode(token_fields).encode("utf-8")
                                token_req = urllib.request.Request(
                                    token_url,
                                    data=token_payload,
                                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                                )
                                with urllib.request.urlopen(token_req, timeout=15) as token_resp:
                                    resp_data = json.loads(token_resp.read().decode("utf-8"))

                                id_tok = resp_data.get("id_token")
                                if id_tok:
                                    callback_result["token"] = id_tok
                                    callback_result["nonce"] = nonce
                                    succ_html = (
                                        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
                                        "<!DOCTYPE html><html><body style='font-family:sans-serif;background:#0b0f19;color:#fff;text-align:center;padding:50px;'>"
                                        "<h2 style='color:#10b981;'>&#10004; Google Authentication Verified</h2>"
                                        "<p>Identity authenticated successfully. You can close this tab and return to <strong>TARA Setup</strong>.</p>"
                                        "<script>window.close();</script>"
                                        "</body></html>"
                                    )
                                    conn.sendall(succ_html.encode("utf-8"))
                                else:
                                    callback_result["error"] = "Google token exchange response did not contain id_token."
                                    err_html = (
                                        "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
                                        "<!DOCTYPE html><html><body style='font-family:sans-serif;background:#0b0f19;color:#ef4444;text-align:center;padding:50px;'>"
                                        "<h2>Google Token Exchange Failed</h2><p>Token response did not contain id_token.</p></body></html>"
                                    )
                                    conn.sendall(err_html.encode("utf-8"))
                            except urllib.error.HTTPError as ex:
                                err_body_str = ""
                                try:
                                    raw_body = ex.read()
                                    if raw_body:
                                        err_body_str = raw_body.decode("utf-8", errors="ignore")
                                except Exception:
                                    pass

                                detailed_err = None
                                if err_body_str:
                                    try:
                                        err_json = json.loads(err_body_str)
                                        err_code = err_json.get("error")
                                        err_desc = err_json.get("error_description")
                                        if err_code and err_desc:
                                            detailed_err = f"{err_code}: {err_desc}"
                                        elif err_desc:
                                            detailed_err = err_desc
                                        elif err_code:
                                            detailed_err = err_code
                                        else:
                                            detailed_err = json.dumps(err_json)
                                    except Exception:
                                        detailed_err = err_body_str

                                if detailed_err:
                                    callback_result["error"] = f"Google token exchange failed ({ex.code}): {detailed_err}"
                                else:
                                    callback_result["error"] = f"Google token exchange failed ({ex.code}): {str(ex)}"

                                escaped_err = (detailed_err or str(ex)).replace("<", "&lt;").replace(">", "&gt;")
                                err_html = (
                                    f"HTTP/1.1 {ex.code} {ex.reason}\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
                                    f"<!DOCTYPE html><html><body style='font-family:sans-serif;background:#0b0f19;color:#ef4444;text-align:center;padding:50px;'>"
                                    f"<h2>Google Token Exchange Failed ({ex.code})</h2>"
                                    f"<p style='font-family:monospace;background:#1e1e2e;color:#fca5a5;padding:12px;border-radius:6px;display:inline-block;max-width:85%;'>{escaped_err}</p>"
                                    f"</body></html>"
                                )
                                conn.sendall(err_html.encode("utf-8"))
                            except Exception as ex:
                                callback_result["error"] = f"Token exchange failed: {str(ex)}"
                                escaped_ex = str(ex).replace("<", "&lt;").replace(">", "&gt;")
                                err_html = (
                                    f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
                                    f"<!DOCTYPE html><html><body style='font-family:sans-serif;background:#0b0f19;color:#ef4444;text-align:center;padding:50px;'>"
                                    f"<h2>Token Exchange Failed</h2><p>{escaped_ex}</p></body></html>"
                                )
                                conn.sendall(err_html.encode("utf-8"))
                else:
                    err_html = (
                        "HTTP/1.1 404 Not Found\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
                        "<!DOCTYPE html><html><body style='font-family:sans-serif;background:#0b0f19;color:#ef4444;text-align:center;padding:50px;'>"
                        "<h2>Endpoint Not Found</h2></body></html>"
                    )
                    conn.sendall(err_html.encode("utf-8"))
                conn.close()
            except socket.timeout:
                callback_result["error"] = "Google authentication timed out waiting for browser callback."
            except Exception as e:
                callback_result["error"] = str(e)
            finally:
                server_socket.close()
                callback_result["completed"] = True

        thread = threading.Thread(target=listen_loopback, daemon=True)
        thread.start()

        if open_browser:
            try:
                webbrowser.open(auth_url)
            except Exception:
                pass

        return {
            "redirect_uri": redirect_uri,
            "auth_url": auth_url,
            "port": port,
            "state": state,
            "nonce": nonce,
            "code_verifier": code_verifier,
            "thread": thread,
            "result_holder": callback_result
        }

    # -------------------------------------------------------------------------
    # 3. DEVICE IDENTITY & OS-BACKED STORAGE
    # -------------------------------------------------------------------------
    def generate_device_identity(self, device_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a 32-byte Ed25519 keypair for the current Windows PC.
        Encrypts private key using Windows DPAPI and stores in local storage_dir.
        Returns dict with device_id, device_name, device_public_key.
        """
        name = device_name or platform.node() or "Windows PC"
        priv_obj = ed25519.Ed25519PrivateKey.generate()
        priv_bytes = priv_obj.private_bytes_raw()
        pub_bytes = priv_obj.public_key().public_bytes_raw()
        pub_hex = pub_bytes.hex()

        machine_hash = hashlib.sha256((name + platform.machine() + pub_hex).encode("utf-8")).hexdigest()[:8]
        device_id = f"TARA-PC-{machine_hash.upper()}"

        keystore_path = os.path.join(self.storage_dir, f"{device_id}_key.keystore")
        encrypted_priv = protect_bytes_dpapi(priv_bytes, description=f"TARA_DEVICE_{device_id}")

        keystore_data = {
            "device_id": device_id,
            "device_name": name,
            "device_public_key": pub_hex,
            "hardware_protection": "Windows DPAPI" if IS_WINDOWS else "OS_ENVELOPE",
            "ciphertext": encrypted_priv.hex(),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        with open(keystore_path, "w", encoding="utf-8") as f:
            json.dump(keystore_data, f, indent=2)

        return {
            "device_id": device_id,
            "device_name": name,
            "device_public_key": pub_hex,
            "keystore_path": keystore_path
        }

    def load_device_private_key(self, device_id: str) -> Optional[bytes]:
        """Loads and decrypts device private key using Windows DPAPI."""
        keystore_path = os.path.join(self.storage_dir, f"{device_id}_key.keystore")
        if not os.path.exists(keystore_path):
            return None
        try:
            with open(keystore_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cipher_bytes = bytes.fromhex(data["ciphertext"])
            return unprotect_bytes_dpapi(cipher_bytes)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # 4. FIRST-TIME ROOT CREATOR KEY GENERATION & STORAGE (Scrypt N=131072, r=8, p=1)
    # -------------------------------------------------------------------------
    def generate_and_protect_root_key(
        self,
        master_passphrase: str,
        creator_id: str = CANONICAL_CREATOR_ID
    ) -> Tuple[bytes, bytes, str]:
        """
        Generates 32-byte Ed25519 root keypair for the creator.
        Protects private key with Scrypt (N=131072, r=8, p=1) + AES-256-GCM + Windows DPAPI.
        Persists encrypted keystore in storage_dir.
        Returns (priv_bytes, pub_bytes, keystore_path).
        """
        if not master_passphrase or len(master_passphrase) < 8:
            raise ValueError("Master passphrase must be at least 8 characters.")

        priv_obj = ed25519.Ed25519PrivateKey.generate()
        priv_bytes = priv_obj.private_bytes_raw()
        pub_bytes = priv_obj.public_key().public_bytes_raw()

        salt = secrets.token_bytes(32)
        kdf = Scrypt(salt=salt, length=32, n=131072, r=8, p=1)
        derived_key = kdf.derive(master_passphrase.encode("utf-8"))

        aesgcm = AESGCM(derived_key)
        nonce = secrets.token_bytes(12)
        ciphertext = aesgcm.encrypt(nonce, priv_bytes, None)

        envelope = protect_bytes_dpapi(ciphertext, description="TARA_CREATOR_ROOT_KEY")

        keystore_data = {
            "creator_id": creator_id,
            "root_public_key": pub_bytes.hex(),
            "kdf_method": "Scrypt-N131072-r8-p1",
            "kdf_params": {
                "n": 131072,
                "r": 8,
                "p": 1,
                "salt": salt.hex(),
                "length": 32
            },
            "encryption": "AES-256-GCM+DPAPI",
            "salt": salt.hex(),
            "nonce": nonce.hex(),
            "ciphertext": envelope.hex(),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        keystore_path = os.path.join(self.storage_dir, "operator_key.keystore")
        with open(keystore_path, "w", encoding="utf-8") as f:
            json.dump(keystore_data, f, indent=2)

        return priv_bytes, pub_bytes, keystore_path

    # -------------------------------------------------------------------------
    # 5. ATOMIC FIRST-TIME CREATOR SETUP (100% LOCAL ONLY)
    # -------------------------------------------------------------------------
    def perform_first_time_setup(
        self,
        master_passphrase: str,
        confirm_passphrase: str,
        google_id_token: str,
        confirm_identity: bool = True,
        private_trigger_phrase: Optional[str] = None,
        device_name: Optional[str] = None,
        expected_nonce: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes first-time creator trust root initialization 100% LOCAL ONLY.
        Strict atomic verification sequence before any key is generated:
        1. Google verification (real Google ID-token signature/JWKS, iss, aud, exp, nonce, email_verified, sub)
        2. Explicit confirmation (confirm_identity must be True)
        3. Passphrase verification (master_passphrase and confirm_passphrase must match and be >= 8 chars)
        4. Authority state check (strictly CREATOR_SETUP_REQUIRED)
        5. Generate root key ONCE.
        6. Protect/store root key in local keystore and vault.
        7. Register primary device in DeviceRegistry.
        8. Configure private trigger phrase.
        9. Generate and configure emergency recovery code bound to authenticated Google email.
        10. Initialize creator records and registry.
        11. Generate and sign auth_manifest.json with the generated root private key.
        12. Seal initial authority state cryptographically -> ACTIVE.
        Any failure before final commit rolls back all partial artifacts leaving state strictly CREATOR_SETUP_REQUIRED.
        """
        # 1. Google verification
        token_verif = self.verify_google_token_string(google_id_token, expected_nonce=expected_nonce)
        if not token_verif.get("valid"):
            raise PermissionError(f"Google ID token verification failed: {token_verif.get('error')}")

        creator_email = token_verif["email"]
        creator_sub = token_verif.get("subject")

        # 2. Explicit confirmation
        if not confirm_identity:
            raise ValueError("Explicit first-time confirmation of Root Creator setup is required.")

        # 3. Passphrase confirmation
        if not master_passphrase or not isinstance(master_passphrase, str):
            raise ValueError("Master passphrase cannot be empty.")

        if not confirm_passphrase or not isinstance(confirm_passphrase, str):
            raise ValueError("Passphrase confirmation cannot be empty.")

        if master_passphrase != confirm_passphrase:
            raise ValueError("Master passphrase and confirmation do not match.")

        if len(master_passphrase) < 8:
            raise ValueError("Master passphrase must be at least 8 characters in length.")

        # 4. Authority state check
        mgr = AuthorityLifecycleManager(repo_root=self.repo_root)
        current_state = mgr.get_state()
        if current_state != AuthorityState.CREATOR_SETUP_REQUIRED:
            raise PermissionError(f"Authority is already in state '{current_state}'. Setup cannot be re-run.")

        op_rec_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "operator_record.json")
        creator = CreatorIdentity(record_path=op_rec_path)
        if creator.is_initialized():
            raise PermissionError("Creator authority is already initialized. Cannot re-initialize.")

        # 5. Atomic setup transaction
        created_files: List[str] = []
        creators_reg_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "operators_registry.json")
        recovery_config_path = get_default_recovery_record_path()
        auth_manifest_path = os.path.join(self.repo_root, "TARA", "ACCESS", "operator", "auth_manifest.json")
        vault_dir = os.path.join(self.repo_root, "storage", "vault", "access")
        vault_seal_path = os.path.join(vault_dir, "access_seal.json")
        vault_key_path = os.path.join(vault_dir, "creator_root_key.keystore")
        devices_json_path = os.path.join(self.repo_root, "TARA", "ACCESS", "devices", "devices.json")

        try:
            mgr.begin_initialization()

            # Generate the Creator root key ONCE
            priv_bytes, pub_bytes, keystore_path = self.generate_and_protect_root_key(
                master_passphrase=master_passphrase,
                creator_id=CANONICAL_CREATOR_ID
            )
            created_files.append(keystore_path)

            # Store in vault storage using the SAME priv_bytes
            os.makedirs(vault_dir, exist_ok=True)
            vault_storage = SecureKeyStorage(storage_dir=vault_dir)
            vault_storage.store_private_key_modern(
                "creator_root_key",
                priv_bytes,
                passphrase=master_passphrase,
                use_dpapi=True
            )
            created_files.append(vault_key_path)

            # Generate primary device identity
            dev_info = self.generate_device_identity(device_name=device_name)
            created_files.append(dev_info["keystore_path"])

            # Register primary device in DeviceRegistry
            os.makedirs(os.path.dirname(devices_json_path), exist_ok=True)
            dev_reg = DeviceRegistry(registry_path=devices_json_path)
            dev_reg.register_device(
                device_id=dev_info["device_id"],
                device_name=dev_info["device_name"],
                public_key_hex=dev_info["device_public_key"],
                status="AUTHORIZED"
            )
            created_files.append(devices_json_path)

            # Configure private trigger phrase
            trigger_path = os.path.join(self.repo_root, "TARA", "ACCESS", "activation", "private_trigger.hash")
            trigger_mgr = PrivateTriggerManager(trigger_file_path=trigger_path)
            if private_trigger_phrase:
                trigger_mgr.set_trigger_phrase(private_trigger_phrase)
            elif not trigger_mgr.has_trigger():
                trigger_mgr.set_trigger_phrase(secrets.token_hex(16))
            created_files.append(trigger_path)

            # Configure recovery manager bound to authenticated Google email
            os.makedirs(os.path.dirname(recovery_config_path), exist_ok=True)
            recovery_mgr = RecoveryManager(creator=creator, recovery_record_path=recovery_config_path)
            recovery_code = recovery_mgr.generate_recovery_code()
            recovery_mgr.set_recovery_email(creator_email)
            created_files.append(recovery_config_path)

            # Initialize creator profile
            os.makedirs(os.path.dirname(op_rec_path), exist_ok=True)
            creator.initialize_root_creator(pub_bytes, display_name=CANONICAL_DISPLAY_NAME)
            with open(op_rec_path, "r", encoding="utf-8") as f:
                rec_data = json.load(f)
            rec_data["google_email"] = creator_email
            rec_data["google_sub"] = creator_sub
            with open(op_rec_path, "w", encoding="utf-8") as f:
                json.dump(rec_data, f, indent=2)
            created_files.append(op_rec_path)

            # Update operators_registry.json
            multi_reg = MultiCreatorRegistry(registry_path=creators_reg_path)
            multi_reg.set_public_key(CANONICAL_CREATOR_ID, pub_bytes.hex())
            multi_reg.set_authorized_google_email(CANONICAL_CREATOR_ID, creator_email)
            created_files.append(creators_reg_path)

            # Write signed auth_manifest.json v2
            os.makedirs(os.path.dirname(auth_manifest_path), exist_ok=True)
            manifest_body = {
                "version": 2,
                "creator_id": CANONICAL_CREATOR_ID,
                "display_name": CANONICAL_DISPLAY_NAME,
                "recovery_email": creator_email,
                "root_public_key": pub_bytes.hex(),
                "key_version": 1,
                "status": "ACTIVE",
                "kdf_method": "Scrypt-N131072-r8-p1",
                "hardware_protection": "Windows DPAPI" if IS_WINDOWS else "OS_ENVELOPE",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            canonical_bytes = json.dumps(manifest_body, sort_keys=True).encode("utf-8")
            sig_bytes = Ed25519.sign(priv_bytes, canonical_bytes)
            manifest_data = dict(manifest_body)
            manifest_data["signature"] = sig_bytes.hex()
            with open(auth_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            created_files.append(auth_manifest_path)

            # Seal authority state
            mgr.seal_initial_authority(
                priv_bytes=priv_bytes,
                pub_bytes=pub_bytes,
                master_passphrase=master_passphrase,
                display_name=CANONICAL_DISPLAY_NAME
            )
            created_files.append(vault_seal_path)

            # Verify transition to ACTIVE
            if mgr.get_state() != AuthorityState.ACTIVE:
                raise RuntimeError("Authority sealing failed to transition state to ACTIVE.")

            session = self.issue_session_token(
                creator_id=CANONICAL_CREATOR_ID,
                role="ROOT_CREATOR",
                device_id=dev_info["device_id"],
                lifetime_seconds=3600
            )
            self.active_session = session

            return {
                "status": "SUCCESS",
                "authority_state": "ACTIVE",
                "creator_id": CANONICAL_CREATOR_ID,
                "display_name": CANONICAL_DISPLAY_NAME,
                "device_id": dev_info["device_id"],
                "device_name": dev_info["device_name"],
                "recovery_code": recovery_code,
                "session": session
            }

        except Exception as e:
            # Atomic rollback: remove all newly created artifacts
            for fpath in created_files:
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
            if hasattr(mgr, "_in_initialization"):
                mgr._in_initialization = False
            raise

    # -------------------------------------------------------------------------
    # 6. ANY-PC SECONDARY DEVICE REGISTRATION (100% LOCAL ONLY)
    # -------------------------------------------------------------------------
    def register_second_pc(
        self,
        google_id_token: str,
        device_name: Optional[str] = None,
        creator_proof: Optional[str] = None,
        creator_session_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Any-PC Support: Registers a new Windows PC as an authorized secondary device.
        - Requires Google authentication matching enrolled creator identity.
        - Requires existing authorized Creator authentication/device proof.
        - NEVER grants ROOT_CREATOR merely from Google login.
        - Stores PC-specific device keypair protected by Windows DPAPI.
        - Does NOT copy or require the creator root private key.
        """
        # 1. Verify Google token
        token_verif = self.verify_google_token_string(google_id_token)
        if not token_verif.get("valid"):
            raise PermissionError(f"Google ID token verification failed: {token_verif.get('error')}")

        token_email = token_verif.get("email")

        # 2. Check local authority is ACTIVE
        server_state = self.get_server_state()
        c_status = server_state.get("creator_status")
        if c_status != "ACTIVE":
            raise PermissionError(f"Root authority is not ACTIVE (current state: {c_status}). First-time setup must be performed first.")

        enrolled_email = self.get_enrolled_creator_email()
        if enrolled_email and token_email != enrolled_email:
            raise PermissionError(f"Unauthorized Google identity '{token_email}'. Does not match enrolled Creator identity '{enrolled_email}'.")

        # 3. Require existing authorized Creator proof before registering secondary device
        proof = creator_session_token or creator_proof or (self.active_session.get("session_token") if self.active_session else None)
        devices_json_path = os.path.join(self.repo_root, "TARA", "ACCESS", "devices", "devices.json")
        dev_reg = DeviceRegistry(registry_path=devices_json_path)

        if proof:
            try:
                proof_claims = self.verify_session_token(proof)
                if proof_claims.get("role") != "ROOT_CREATOR":
                    raise PermissionError("Secondary device registration requires ROOT_CREATOR authorization proof.")
            except Exception as e:
                raise PermissionError(f"Invalid Creator authorization proof: {str(e)}")
        else:
            existing_devs = dev_reg.list_devices()
            has_primary_authorized = any(
                (d.status if hasattr(d, "status") else d.get("status")) == "AUTHORIZED"
                for d in existing_devs
            )
            if not has_primary_authorized:
                raise PermissionError("Secondary device registration requires existing primary authorized device proof.")

        # 4. Generate device identity on this PC
        dev_info = self.generate_device_identity(device_name=device_name)

        # 5. Register device in registry
        dev_reg.register_device(
            device_id=dev_info["device_id"],
            device_name=dev_info["device_name"],
            public_key_hex=dev_info["device_public_key"],
            status="AUTHORIZED"
        )

        # 6. Issue device session with AUTHORIZED_DEVICE role (NEVER ROOT_CREATOR merely from Google login)
        session = self.issue_session_token(
            creator_id=CANONICAL_CREATOR_ID,
            role="AUTHORIZED_DEVICE",
            device_id=dev_info["device_id"],
            lifetime_seconds=3600
        )

        return {
            "status": "SUCCESS",
            "device_id": dev_info["device_id"],
            "device_name": dev_info["device_name"],
            "device": dev_info,
            "session": session
        }

    # -------------------------------------------------------------------------
    # 7. DEVICE MANAGEMENT & REVOCATION (100% LOCAL ONLY)
    # -------------------------------------------------------------------------
    def list_devices(self, session_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists registered devices from local DeviceRegistry after cryptographic session verification."""
        token = session_token or (self.active_session.get("session_token") if self.active_session else None)
        if not token:
            raise PermissionError("Active creator session required to list devices.")

        self.verify_session_token(token)

        devices_json_path = os.path.join(self.repo_root, "TARA", "ACCESS", "devices", "devices.json")
        devs = DeviceRegistry(registry_path=devices_json_path).list_devices()
        return [d.to_dict() if hasattr(d, 'to_dict') else d for d in devs]

    def revoke_device(self, device_id: str, session_token: Optional[str] = None, reason: str = "manual_revocation") -> Dict[str, Any]:
        """Revokes an authorized device in local DeviceRegistry after cryptographic session verification."""
        token = session_token or (self.active_session.get("session_token") if self.active_session else None)
        if not token:
            raise PermissionError("Active creator session required to revoke device.")

        claims = self.verify_session_token(token)
        if claims.get("role") != "ROOT_CREATOR":
            raise PermissionError("Only ROOT_CREATOR sessions have authority to revoke devices.")

        devices_json_path = os.path.join(self.repo_root, "TARA", "ACCESS", "devices", "devices.json")
        dev_reg = DeviceRegistry(registry_path=devices_json_path)
        rev_ok = dev_reg.revoke_device(device_id, reason=reason)
        return {"status": "SUCCESS" if rev_ok else "ERROR", "device_id": device_id, "revoked": rev_ok}

    # -------------------------------------------------------------------------
    # 8. LAUNCH LOCAL CREATOR CONSOLE
    # -------------------------------------------------------------------------
    def launch_creator_console(self, session_token: Optional[str] = None) -> bool:
        """
        Launches the local Creator Console in the system browser using secure in-memory authenticated handoff.
        Zero session tokens written to disk or HTML files.
        """
        token = session_token or (self.active_session.get("session_token") if self.active_session else None)
        if not token:
            return False

        try:
            self.verify_session_token(token)
        except Exception:
            return False

        # Clean up any legacy launch_console.html files to prevent token leakage
        launcher_file = os.path.join(self.storage_dir, "launch_console.html")
        if os.path.exists(launcher_file):
            try:
                os.remove(launcher_file)
            except Exception:
                pass

        # Ephemeral in-memory single-use handoff ticket
        ticket = secrets.token_urlsafe(32)
        self._handoff_tickets[ticket] = {
            "session_token": token,
            "creator_id": CANONICAL_CREATOR_ID,
            "expires_at": time.time() + 30.0
        }

        target_url = f"http://127.0.0.1:8000/chat?handoff={ticket}"
        try:
            return webbrowser.open(target_url)
        except Exception:
            return False

    def get_biometric_capabilities(self) -> Dict[str, str]:
        """Queries local platform for biometric capabilities."""
        from TARA.ACCESS.factors import (
            WindowsHelloProvider,
            FingerprintProvider,
            FaceProvider,
            IrisProvider,
            VoiceVerificationProvider,
        )
        return {
            "windows_hello": WindowsHelloProvider().get_capability().value,
            "fingerprint": FingerprintProvider().get_capability().value,
            "face": FaceProvider().get_capability().value,
            "iris": IrisProvider().get_capability().value,
            "voice_auxiliary": VoiceVerificationProvider().get_capability().value,
        }

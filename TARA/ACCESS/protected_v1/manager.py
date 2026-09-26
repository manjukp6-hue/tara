"""
TARA/ACCESS/protected_v1/manager.py

Manages lifecycle and state of capability_v1 / protected_state_v1.
Additive to normal ROOT_CREATOR:
- When Inactive: Normal ROOT_CREATOR behavior.
- When Active: Normal ROOT_CREATOR + Self-Control capabilities.

Strict Invariants:
- Private trigger phrase is NEVER stored in plaintext.
- Activation requires BOTH valid authenticated ROOT_CREATOR session AND correct private verifier.
- Automatically deactivates on session logout, expiry, or explicit command.
- No descriptive or forbidden terminology used.
"""

import os
import json
import secrets
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, Set

CAPABILITY_V1 = "capability_v1"
CANONICAL_CREATOR_ID = "ROOT_OPERATOR"
REQUIRED_ROLE = "ROOT_CREATOR"


def normalize_trigger_phrase(text: str) -> str:
    """Normalizes trigger input (lowercase, collapsed spaces, stripped non-alphanumeric)."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(cleaned.split())


class ProtectedStateManager:
    KDF_ITERATIONS = 100_000

    def __init__(
        self,
        trigger_file_path: Optional[str] = None,
        audit_logger: Optional[Any] = None,
        repo_root: Optional[str] = None
    ):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.repo_root = repo_root

        if trigger_file_path is None:
            trigger_file_path = os.path.join(
                self.repo_root, "TARA", "ACCESS", "operator", "state_v1_verifier.json"
            )
        self.trigger_file_path = trigger_file_path
        self.audit_logger = audit_logger

        self._salt: Optional[bytes] = None
        self._hash: Optional[str] = None
        self._updated_at: Optional[str] = None

        # Active sessions: session_token -> metadata
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

        self.load_trigger()

    def has_trigger(self) -> bool:
        return self._salt is not None and self._hash is not None

    def load_trigger(self) -> bool:
        if os.path.exists(self.trigger_file_path):
            try:
                with open(self.trigger_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._salt = bytes.fromhex(data["trigger_salt"])
                self._hash = data["trigger_hash"]
                self._updated_at = data.get("updated_at")
                return True
            except Exception:
                return False
        return False

    def set_trigger_phrase(self, phrase: str) -> Dict[str, Any]:
        """
        Sets a new private verifier using PBKDF2-HMAC-SHA256.
        Plaintext is strictly never saved to disk.
        """
        if not phrase or not isinstance(phrase, str):
            raise ValueError("Trigger phrase cannot be empty.")

        norm = normalize_trigger_phrase(phrase)
        if len(norm) < 8:
            raise ValueError("Trigger phrase must be at least 8 characters in length.")

        salt = secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            norm.encode("utf-8"),
            salt,
            self.KDF_ITERATIONS
        ).hex()

        self._salt = salt
        self._hash = derived
        self._updated_at = datetime.now(timezone.utc).isoformat()

        os.makedirs(os.path.dirname(self.trigger_file_path), exist_ok=True)
        with open(self.trigger_file_path, "w", encoding="utf-8") as f:
            json.dump({
                "trigger_salt": salt.hex(),
                "trigger_hash": derived,
                "kdf": "PBKDF2-HMAC-SHA256",
                "iterations": self.KDF_ITERATIONS,
                "version": "1.0",
                "updated_at": self._updated_at
            }, f, indent=2)

        return {
            "status": "SUCCESS",
            "message": "Protected state verifier established.",
            "updated_at": self._updated_at
        }

    def verify_trigger(self, candidate_text: str) -> bool:
        """
        Verifies whether candidate_text matches the configured PBKDF2 verifier.
        Constant-time comparison used.
        """
        if not self.has_trigger() or not candidate_text:
            return False

        norm = normalize_trigger_phrase(candidate_text)
        if len(norm) < 8:
            return False

        derived = hashlib.pbkdf2_hmac(
            "sha256",
            norm.encode("utf-8"),
            self._salt,
            self.KDF_ITERATIONS
        ).hex()

        return hmac.compare_digest(derived, self._hash)

    def activate(
        self,
        session_token: str,
        candidate_trigger: str,
        creator_auth_service: Any
    ) -> Tuple[bool, str]:
        """
        Dual-factor activation gate:
        1. Verifies valid authenticated ROOT_CREATOR session.
        2. Verifies candidate trigger matches PBKDF2 verifier.
        """
        if not session_token or not candidate_trigger:
            return False, "Access denied: Missing authentication credentials."

        if not creator_auth_service:
            return False, "Access denied: Authentication service unavailable."

        session_info = creator_auth_service.verify_session(session_token)
        if not session_info:
            return False, "Access denied: Invalid or expired creator session."

        creator_id = session_info.get("creator_id")
        role = session_info.get("role")

        if creator_id != CANONICAL_CREATOR_ID or role != REQUIRED_ROLE:
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="state_v1_activation_requested",
                    severity="WARNING",
                    details={"creator_id": creator_id, "role": role, "result": "DENIED_ROLE"}
                )
            return False, "Access denied: Insufficient creator role."

        if not self.verify_trigger(candidate_trigger):
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="state_v1_activation_requested",
                    severity="WARNING",
                    details={"creator_id": creator_id, "result": "DENIED_TRIGGER"}
                )
            return False, "Access denied: Trigger verification failed."

        self._active_sessions[session_token] = {
            "creator_id": creator_id,
            "role": role,
            "capability": CAPABILITY_V1,
            "activated_at": datetime.now(timezone.utc).isoformat()
        }

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="state_v1_activated",
                severity="CRITICAL",
                details={"creator_id": creator_id, "capability": CAPABILITY_V1}
            )

        return True, "✅ Protected capability activated."

    def deactivate(self, session_token: str) -> Tuple[bool, str]:
        """Explicitly deactivates the protected state for the session."""
        if session_token in self._active_sessions:
            info = self._active_sessions.pop(session_token)
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="state_v1_deactivated",
                    severity="INFO",
                    details={"creator_id": info.get("creator_id")}
                )
        return True, "🔒 Protected capability deactivated."

    def is_active(self, session_token: str, creator_auth_service: Optional[Any] = None) -> bool:
        """
        Checks if capability_v1 is active for the given session token.
        Automatically validates that the underlying ROOT_CREATOR session is still alive.
        """
        if not session_token or session_token not in self._active_sessions:
            return False

        if creator_auth_service:
            sess = creator_auth_service.verify_session(session_token)
            if not sess or sess.get("creator_id") != CANONICAL_CREATOR_ID or sess.get("role") != REQUIRED_ROLE:
                self._active_sessions.pop(session_token, None)
                if self.audit_logger:
                    self.audit_logger.log_event(
                        event_type="state_v1_deactivated",
                        severity="INFO",
                        details={"reason": "session_expired_or_revoked"}
                    )
                return False

        return True



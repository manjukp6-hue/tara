"""
TARA/ACCESS/trigger/private_trigger.py

Cryptographically Secured Private Creator Trigger Phrase Manager.
Enforces:
1. Private trigger phrase is initiated conversationally in normal chat.
2. Only salted PBKDF2-HMAC-SHA256 verifiers are stored; plaintext is NEVER stored,
   never logged, never exposed in API responses, audit logs, or model context.
3. Generic messages ("I need creator login", "creator login beku", etc.) must NEVER
   trigger authentication and are rejected even if attempted as triggers.
4. Knowing the trigger phrase alone MUST NEVER grant creator authority;
   it only transitions the state to request verified Google authentication.
"""

import os
import json
import hmac
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Set

GENERIC_LOGIN_PHRASES: Set[str] = {
    "i need creator login",
    "creator login beku",
    "i want creator access",
    "authenticate me as creator",
    "nanage creator authority beku",
    "creator login",
    "login as creator",
    "give me creator access",
    "creator access beku",
    "admin login",
    "login",
    "creator",
    "root login",
    "authenticate",
    "nanu creator",
    "im creator",
    "i am creator",
    "i am the creator"
}


def normalize_phrase(text: str) -> str:
    """Normalizes input text for phrase comparison (lowercase, collapsed spaces, stripped punctuation)."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(cleaned.split())


class PrivateTriggerManager:
    """
    Manages the private creator trigger phrase verifier.
    Stores only salted PBKDF2-HMAC-SHA256 hashes.
    """
    KDF_ITERATIONS = 100_000

    def __init__(self, trigger_file_path: Optional[str] = None):
        if trigger_file_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            trigger_file_path = os.path.join(repo_root, "TARA", "ACCESS", "operator", "activation_config.json")
        self.trigger_file_path = trigger_file_path
        self._salt: Optional[bytes] = None
        self._hash: Optional[str] = None
        self._updated_at: Optional[str] = None
        self.load()

    def has_trigger(self) -> bool:
        """Returns True if a valid trigger hash verifier is configured."""
        return self._salt is not None and self._hash is not None

    def is_generic_phrase(self, text: str) -> bool:
        """Returns True if the text matches a generic, non-private login attempt."""
        norm = normalize_phrase(text)
        return norm in GENERIC_LOGIN_PHRASES

    def set_trigger_phrase(self, phrase: str) -> Dict[str, Any]:
        """
        Sets a new private creator trigger phrase.
        Validates that the phrase is non-generic and sufficiently distinct.
        Computes a salted PBKDF2-HMAC-SHA256 hash and saves the verifier.
        Never persists or logs the raw phrase.
        """
        if not phrase or not isinstance(phrase, str):
            raise ValueError("Trigger phrase cannot be empty.")

        norm = normalize_phrase(phrase)
        if len(norm) < 6:
            raise ValueError("Trigger phrase must be at least 6 characters in length.")

        if norm in GENERIC_LOGIN_PHRASES:
            raise ValueError(
                f"Generic phrase '{phrase}' cannot be used as a private creator trigger. "
                "Choose a unique, personal trigger phrase."
            )

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
        self.save()

        return {
            "status": "SUCCESS",
            "message": "Private creator trigger phrase verifier successfully established.",
            "updated_at": self._updated_at
        }

    def verify_trigger(self, candidate_text: str) -> bool:
        """
        Verifies whether candidate_text matches the configured private trigger phrase.
        Uses constant-time comparison to prevent timing attacks.
        Generic phrases are rejected immediately.
        """
        if not candidate_text or not isinstance(candidate_text, str):
            return False

        norm = normalize_phrase(candidate_text)
        if not norm:
            return False

        # Generic phrases MUST NOT trigger authentication
        if norm in GENERIC_LOGIN_PHRASES:
            return False

        if self._salt is None or self._hash is None:
            return False

        candidate_derived = hashlib.pbkdf2_hmac(
            "sha256",
            norm.encode("utf-8"),
            self._salt,
            self.KDF_ITERATIONS
        ).hex()

        return hmac.compare_digest(candidate_derived, self._hash)

    def rotate_trigger_phrase(self, old_phrase: str, new_phrase: str) -> Dict[str, Any]:
        """
        Rotates the private trigger phrase.
        Requires valid proof of the previous trigger phrase before setting the new one.
        """
        if not self.verify_trigger(old_phrase):
            raise PermissionError("Current trigger phrase verification failed.")
        return self.set_trigger_phrase(new_phrase)

    def save(self) -> None:
        """Persists salt and hash verifier to disk. Never saves raw phrase."""
        if self._salt is None or self._hash is None:
            return
        os.makedirs(os.path.dirname(self.trigger_file_path), exist_ok=True)
        data = {
            "trigger_salt": self._salt.hex(),
            "trigger_hash": self._hash,
            "kdf": "PBKDF2-HMAC-SHA256",
            "iterations": self.KDF_ITERATIONS,
            "updated_at": self._updated_at or datetime.now(timezone.utc).isoformat()
        }
        with open(self.trigger_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Loads salt and hash verifier from disk."""
        if not os.path.exists(self.trigger_file_path):
            return
        try:
            with open(self.trigger_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            salt_hex = data.get("trigger_salt")
            hash_val = data.get("trigger_hash")
            if salt_hex and hash_val:
                self._salt = bytes.fromhex(salt_hex)
                self._hash = hash_val
                self._updated_at = data.get("updated_at")
        except Exception:
            pass

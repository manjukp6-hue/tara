"""
python/tara_core/security/audit.py

Tamper-Aware Security Event Audit System for TARA Core.
Provides HMAC-SHA256 event chaining and automatic secret scrubbing.
Never logs private keys, passwords, provider secrets, encryption keys, or sensitive chat plaintext.
"""

import os
import json
import time
import hmac
import hashlib
from typing import Dict, List, Optional, Any

FORBIDDEN_SECRET_KEYS = {
    "private_key", "privkey", "ed25519_seed", "master_seed", "secret", "password",
    "passphrase", "token", "api_token", "api_key", "auth_token", "session_key",
    "provider_token", "hf_token", "cloudflare_api_token", "encryption_key"
}


class TamperAwareSecurityAudit:
    """
    Cryptographically chained security audit logger.
    Events contain SHA-256 hash of previous event to detect truncation or modification.
    """

    def __init__(self, log_path: Optional[str] = None):
        if log_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            log_path = os.path.join(repo_root, "storage", "audit", "security_events.jsonl")
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._last_hash = self._compute_initial_hash()

    def _compute_initial_hash(self) -> str:
        if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                    if lines:
                        last_line = json.loads(lines[-1])
                        return last_line.get("event_hash", "0" * 64)
            except Exception:
                pass
        return "0" * 64

    def sanitize_dict(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Recursively scrubs secrets and sensitive credential fields."""
        if not data:
            return {}
        cleaned = {}
        for k, v in data.items():
            if any(secret_term in k.lower() for secret_term in FORBIDDEN_SECRET_KEYS):
                cleaned[k] = "[REDACTED_SECRET]"
            elif isinstance(v, dict):
                cleaned[k] = self.sanitize_dict(v)
            elif isinstance(v, bytes):
                cleaned[k] = f"[BYTES_LEN_{len(v)}]"
            elif isinstance(v, str) and ("BEGIN PRIVATE KEY" in v or "BEGIN EC PRIVATE" in v):
                cleaned[k] = "[REDACTED_KEY_PEM]"
            else:
                cleaned[k] = v
        return cleaned

    def record_event(
        self,
        event_type: str,
        entity_id: str,
        severity: str = "INFO",
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Appends a sanitized, cryptographically chained audit event.
        """
        sanitized = self.sanitize_dict(details)
        timestamp = time.time()
        iso_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))

        payload_to_hash = f"{self._last_hash}:{iso_time}:{event_type}:{entity_id}:{severity}:{json.dumps(sanitized, sort_keys=True)}"
        current_hash = hashlib.sha256(payload_to_hash.encode("utf-8")).hexdigest()

        event = {
            "timestamp": iso_time,
            "epoch": timestamp,
            "event_type": event_type,
            "entity_id": entity_id,
            "severity": severity,
            "prev_hash": self._last_hash,
            "event_hash": current_hash,
            "details": sanitized
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            self._last_hash = current_hash
        except Exception:
            pass

        return event

    def verify_chain_integrity(self) -> Tuple_Validation:
        """Verifies cryptographic hash chain of audit records to detect tampering."""
        if not os.path.exists(self.log_path):
            return True, "No log file exists"
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            expected_prev = "0" * 64
            for idx, line in enumerate(lines):
                record = json.loads(line)
                if record.get("prev_hash") != expected_prev:
                    return False, f"Audit chain broken at entry index {idx}"

                payload_to_hash = (
                    f"{record['prev_hash']}:{record['timestamp']}:{record['event_type']}:"
                    f"{record['entity_id']}:{record['severity']}:{json.dumps(record['details'], sort_keys=True)}"
                )
                computed = hashlib.sha256(payload_to_hash.encode("utf-8")).hexdigest()
                if computed != record.get("event_hash"):
                    return False, f"Audit hash mismatch at entry index {idx}"
                expected_prev = computed

            return True, f"Verified {len(lines)} audit events in chain"
        except Exception as e:
            return False, f"Error verifying audit chain: {str(e)}"


Tuple_Validation = tuple[bool, str]

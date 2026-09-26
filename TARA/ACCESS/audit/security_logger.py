"""
TARA/ACCESS/audit/security_logger.py

Non-Secret Security Audit Logger for TARA with Cryptographic Hash Chaining.
Records critical security events:
- AUTH_FAILURE
- LOCKOUT
- LOCKDOWN_ENGAGED
- RECOVERY_INITIATED
- RECOVERY_SUCCESS
- DEVICE_AUTHORIZED
- KEY_ROTATED
- KEY_REVOKED
- DESTRUCTION_ARMED
- DESTRUCTION_COMPLETED

Strict Invariant:
NEVER logs private keys, recovery codes, passwords, biometric templates, or raw tokens.
All log entries are linked in a tamper-evident SHA-256 hash chain.
"""

import os
import json
import secrets
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple


FORBIDDEN_KEYS = {
    "private_key", "priv_key", "priv_bytes", "private_bytes",
    "recovery_code", "recovery_secret", "raw_code", "password",
    "passphrase", "biometric_template", "token", "raw_token",
    "device_private_key", "creator_private_key"
}

GENESIS_HASH = "0" * 64


class SecurityAuditLogger:
    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            log_dir = os.path.join(repo_root, "TARA", "ACCESS", "audit")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "security_audit.jsonl")
        self._in_memory_events: List[Dict[str, Any]] = []
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        """Loads the entry_hash of the last logged event, or returns GENESIS_HASH."""
        if os.path.exists(self.log_file):
            try:
                last_line = None
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            last_line = line.strip()
                if last_line:
                    data = json.loads(last_line)
                    return data.get("entry_hash", GENESIS_HASH)
            except Exception:
                pass
        return GENESIS_HASH

    def _sanitize(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitizes details to ensure no sensitive secret material is logged."""
        sanitized = {}
        for k, v in details.items():
            k_lower = k.lower()
            if any(forbidden in k_lower for forbidden in FORBIDDEN_KEYS):
                sanitized[k] = "[REDACTED_SECRET]"
            elif isinstance(v, dict):
                sanitized[k] = self._sanitize(v)
            elif isinstance(v, bytes):
                sanitized[k] = f"<bytes len={len(v)}>"
            else:
                sanitized[k] = v
        return sanitized

    def _compute_entry_hash(
        self,
        prev_hash: str,
        timestamp: str,
        event_id: str,
        event_type: str,
        severity: str,
        source_device: Optional[str],
        details: Dict[str, Any]
    ) -> str:
        """Computes cryptographic SHA-256 digest for an audit entry."""
        payload = (
            f"{prev_hash}|{timestamp}|{event_id}|{event_type}|"
            f"{severity}|{source_device or ''}|{json.dumps(details, sort_keys=True)}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def log_event(
        self,
        event_type: str,
        severity: str = "INFO",
        details: Optional[Dict[str, Any]] = None,
        source_device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Records a security audit event with cryptographic hash chaining."""
        sanitized_details = self._sanitize(details or {})
        timestamp = datetime.now(timezone.utc).isoformat()
        event_id = secrets.token_hex(8)
        prev_hash = self._last_hash

        entry_hash = self._compute_entry_hash(
            prev_hash=prev_hash,
            timestamp=timestamp,
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            source_device=source_device,
            details=sanitized_details
        )

        event = {
            "timestamp": timestamp,
            "event_id": event_id,
            "event_type": event_type,
            "severity": severity,
            "source_device": source_device,
            "details": sanitized_details,
            "prev_hash": prev_hash,
            "entry_hash": entry_hash
        }

        self._in_memory_events.append(event)
        self._last_hash = entry_hash

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass
        return event

    def verify_log_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Cryptographically verifies the audit log hash chain.
        Returns (True, None) if completely intact, or (False, error_reason).
        Supports backward compatibility with legacy entries before hash chaining.
        """
        if not os.path.exists(self.log_file):
            return True, None

        expected_prev_hash = GENESIS_HASH
        line_idx = 0

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    line_idx += 1
                    try:
                        entry = json.loads(line_str)
                    except Exception as e:
                        return False, f"Invalid JSON on line {line_idx}: {str(e)}"

                    actual_entry_hash = entry.get("entry_hash")
                    if not actual_entry_hash:
                        # Legacy unchained entry prior to hash-chaining hardening
                        continue

                    actual_prev_hash = entry.get("prev_hash")
                    if not actual_prev_hash or not hmac.compare_digest(actual_prev_hash, expected_prev_hash):
                        return False, (
                            f"Broken chain link at line {line_idx} (event {entry.get('event_id')}): "
                            f"expected prev_hash {expected_prev_hash}, got {actual_prev_hash}"
                        )

                    expected_entry_hash = self._compute_entry_hash(
                        prev_hash=entry.get("prev_hash", ""),
                        timestamp=entry.get("timestamp", ""),
                        event_id=entry.get("event_id", ""),
                        event_type=entry.get("event_type", ""),
                        severity=entry.get("severity", ""),
                        source_device=entry.get("source_device"),
                        details=entry.get("details", {})
                    )
                    if not hmac.compare_digest(expected_entry_hash, actual_entry_hash):
                        return False, (
                            f"Tampered entry content at line {line_idx} (event {entry.get('event_id')}): "
                            f"computed hash {expected_entry_hash} != stored hash {actual_entry_hash}"
                        )

                    expected_prev_hash = actual_entry_hash

            return True, None
        except Exception as e:
            return False, f"Failed to read audit log: {str(e)}"

    def get_events(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves non-secret audit events."""
        events = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            e = json.loads(line)
                            if event_type is None or e.get("event_type") == event_type:
                                events.append(e)
            except Exception:
                events = list(self._in_memory_events)
        else:
            events = [e for e in self._in_memory_events if event_type is None or e.get("event_type") == event_type]
        return events[-limit:]

    def shred_and_delete(self) -> bool:
        """Cryptographically shreds the audit log file during total self-destruct."""
        self._in_memory_events.clear()
        self._last_hash = GENESIS_HASH
        if os.path.exists(self.log_file):
            try:
                size = os.path.getsize(self.log_file)
                with open(self.log_file, "wb") as f:
                    f.write(secrets.token_bytes(max(size, 1024)))
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass
                os.remove(self.log_file)
                return True
            except Exception:
                return False
        return True

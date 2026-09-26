"""
TARA/ACCESS/lockdown/lockdown_manager.py

Progressive and Full Lockdown Engine for TARA Security System.
Controls system security states:
  NORMAL -> LOCKDOWN -> RECOVERY -> DESTROYING -> DESTROYED

Rules:
1. Invalid auth -> DENY -> Audit Event.
2. 3 consecutive malicious failures -> Temporary Lockout.
3. 5 consecutive malicious failures -> FULL LOCKDOWN.
4. Accidental biometric failures != malicious cryptographic attacks (isolated tracking).
5. In FULL LOCKDOWN:
   - Creator commands BLOCKED.
   - Identity modification BLOCKED.
   - Key rotation BLOCKED.
   - Device authorization BLOCKED.
   - Protected skill / model administration BLOCKED.
   - Recovery initiation ALLOWED.
   - Safe lockdown status query ALLOWED.
   - Google or Firebase availability CANNOT bypass lockdown.
6. Creator absence (app closed, offline, network loss) NEVER triggers destruction.
"""

import os
import json
import time
from enum import Enum
from typing import Dict, Any, Optional, List

from ..audit.security_logger import SecurityAuditLogger


class SecurityState(str, Enum):
    NORMAL = "NORMAL"
    LOCKDOWN = "LOCKDOWN"
    RECOVERY = "RECOVERY"
    DESTROYING = "DESTROYING"
    DESTROYED = "DESTROYED"


class LockdownManager:
    """
    Coordinates progressive attack mitigation and full lockdown state.
    """
    TEMPORARY_LOCK_THRESHOLD = 3
    FULL_LOCKDOWN_THRESHOLD = 5
    TEMPORARY_LOCK_DURATION_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        state_file: Optional[str] = None,
        audit_logger: Optional[SecurityAuditLogger] = None
    ):
        if state_file is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            state_file = os.path.join(repo_root, "TARA", "ACCESS", "lockdown", "lockdown_state.json")
        self.state_file = state_file
        self.audit_logger = audit_logger or SecurityAuditLogger()

        self.current_state: SecurityState = SecurityState.NORMAL
        self.crypto_failure_count: int = 0
        self.biometric_failure_count: int = 0
        self.temporary_locked_until: Optional[float] = None
        self.lockdown_reason: Optional[str] = None
        self.write_block_active: bool = False

        if os.path.exists(self.state_file):
            self.load()

    def is_temporarily_locked(self) -> bool:
        """Checks if temporary rate-limiting lockout is active."""
        if self.temporary_locked_until is None:
            return False
        now = time.time()
        if now < self.temporary_locked_until:
            return True
        # Lockout expired
        self.temporary_locked_until = None
        self.save()
        return False

    def is_in_lockdown(self) -> bool:
        """Checks if system is in full LOCKDOWN, DESTROYING, or DESTROYED state."""
        return self.current_state in (SecurityState.LOCKDOWN, SecurityState.DESTROYING, SecurityState.DESTROYED)

    def record_crypto_failure(self, details: Dict[str, Any], source_device: Optional[str] = None) -> Dict[str, Any]:
        """
        Records a malicious cryptographic failure (invalid signature, wrong key, tampered message).
        Triggers progressive temporary lock at 3 failures, full lockdown at 5 failures.
        """
        if self.current_state in (SecurityState.DESTROYING, SecurityState.DESTROYED):
            return {"state": self.current_state.value, "action": "DENIED"}

        self.crypto_failure_count += 1
        self.audit_logger.log_event(
            "AUTH_FAILURE",
            severity="WARNING",
            details={"crypto_failures": self.crypto_failure_count, **details},
            source_device=source_device
        )

        action = "DENY"
        if self.crypto_failure_count >= self.FULL_LOCKDOWN_THRESHOLD:
            self.current_state = SecurityState.LOCKDOWN
            self.lockdown_reason = f"FULL LOCKDOWN engaged: {self.crypto_failure_count} consecutive cryptographic attack failures."
            self.audit_logger.log_event(
                "LOCKDOWN_ENGAGED",
                severity="CRITICAL",
                details={"reason": self.lockdown_reason, "failures": self.crypto_failure_count}
            )
            action = "ENGAGE_FULL_LOCKDOWN"
        elif self.crypto_failure_count >= self.TEMPORARY_LOCK_THRESHOLD:
            self.temporary_locked_until = time.time() + self.TEMPORARY_LOCK_DURATION_SECONDS
            self.audit_logger.log_event(
                "TEMPORARY_LOCKOUT",
                severity="HIGH",
                details={"locked_until": self.temporary_locked_until, "failures": self.crypto_failure_count}
            )
            action = "TEMPORARY_LOCKOUT"

        self.save()
        return {
            "state": self.current_state.value,
            "action": action,
            "failures": self.crypto_failure_count,
            "temporarily_locked": self.is_temporarily_locked()
        }

    def record_biometric_failure(self, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Records an accidental biometric mismatch.
        In accordance with rule 12, biometric misreads do NOT count as malicious cryptographic attacks
        and do NOT automatically trigger full lockdown.
        """
        self.biometric_failure_count += 1
        self.audit_logger.log_event(
            "BIOMETRIC_FAIL",
            severity="INFO",
            details={"biometric_failures": self.biometric_failure_count, **(details or {})}
        )
        return {
            "state": self.current_state.value,
            "action": "DEVICE_PIN_FALLBACK_REQUIRED",
            "biometric_failures": self.biometric_failure_count
        }

    def record_success(self) -> None:
        """Resets failure counters on successful authenticated creator operation."""
        if self.current_state == SecurityState.NORMAL:
            self.crypto_failure_count = 0
            self.biometric_failure_count = 0
            self.temporary_locked_until = None
            self.save()

    def engage_manual_lockdown(self, reason: str = "Manual creator security lockdown") -> None:
        """Explicitly engages full lockdown."""
        if self.current_state in (SecurityState.DESTROYING, SecurityState.DESTROYED):
            return
        self.current_state = SecurityState.LOCKDOWN
        self.lockdown_reason = reason
        self.audit_logger.log_event("LOCKDOWN_ENGAGED", severity="CRITICAL", details={"reason": reason})
        self.save()

    def clear_lockdown_via_recovery(self) -> bool:
        """Clears lockdown following successful authorized recovery."""
        if self.current_state in (SecurityState.DESTROYING, SecurityState.DESTROYED):
            return False  # Irreversible after destruction!
        self.current_state = SecurityState.NORMAL
        self.crypto_failure_count = 0
        self.biometric_failure_count = 0
        self.temporary_locked_until = None
        self.lockdown_reason = None
        self.audit_logger.log_event("LOCKDOWN_CLEARED", severity="INFO", details={"method": "recovery"})
        self.save()
        return True

    def can_execute_privileged_operation(self, operation_name: str) -> bool:
        """
        Checks whether an operation is permitted under current lockdown state.
        BLOCKED in LOCKDOWN:
        - creator commands
        - identity modification
        - key rotation
        - device authorization
        - self-destruct arming
        - protected skill / model administration
        ALLOWED in LOCKDOWN:
        - status query
        - non-sensitive diagnostics
        - recovery initiation
        """
        if self.current_state in (SecurityState.DESTROYING, SecurityState.DESTROYED):
            return False

        if self.is_in_lockdown() or self.is_temporarily_locked():
            allowed_in_lockdown = {"status", "get_lockdown_status", "get_status", "initiate_recovery", "verify_recovery"}
            return operation_name.lower() in allowed_in_lockdown

        return True

    def set_destroying_state(self) -> None:
        """Transitions to irreversible DESTROYING state and activates persistent write block."""
        self.current_state = SecurityState.DESTROYING
        self.write_block_active = True
        self.lockdown_reason = "SYSTEM_DESTROYING"
        self.save()

    def set_destroyed_state(self) -> None:
        """Transitions to terminal DESTROYED state. No further operations permitted."""
        self.current_state = SecurityState.DESTROYED
        self.write_block_active = True
        self.lockdown_reason = "SYSTEM_DESTROYED"
        self.save()

    def is_write_blocked(self) -> bool:
        """Checks whether persistent writes are frozen."""
        return self.write_block_active or self.current_state in (SecurityState.DESTROYING, SecurityState.DESTROYED)

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {
                "current_state": self.current_state.value,
                "crypto_failure_count": self.crypto_failure_count,
                "biometric_failure_count": self.biometric_failure_count,
                "temporary_locked_until": self.temporary_locked_until,
                "lockdown_reason": self.lockdown_reason,
                "write_block_active": self.write_block_active
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load(self) -> None:
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.current_state = SecurityState(data.get("current_state", SecurityState.NORMAL.value))
            self.crypto_failure_count = data.get("crypto_failure_count", 0)
            self.biometric_failure_count = data.get("biometric_failure_count", 0)
            self.temporary_locked_until = data.get("temporary_locked_until")
            self.lockdown_reason = data.get("lockdown_reason")
            self.write_block_active = data.get("write_block_active", False)
        except Exception:
            pass

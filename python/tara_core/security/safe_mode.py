"""
python/tara_core/security/safe_mode.py

Externally Enforced TARA Safe Mode Architecture.
If TARA or its runtime exhibits signs of compromise or high anomaly:
1. Enters restricted Safe Mode
2. Permitted: health, diagnostics, read-only status, security reporting, approved recovery, controlled user communication
3. Disabled: privileged self-modification, model promotion, skill promotion, protected capability activation,
   provider credential access, destructive actions, unrestricted tool execution, arbitrary filesystem writes,
   arbitrary network access, uncontrolled knowledge promotion.
4. Non-override Invariant: The model ITSELF can NEVER exit safe mode.
   Only an external cryptographic creator token or independent watchdog can deactivate safe mode.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID

logger = logging.getLogger("tara_core.security.safe_mode")


@dataclass
class SafeModeStatus:
    is_active: bool = False
    triggered_by: str = "SYSTEM"
    reason: Optional[str] = None
    activated_at: Optional[float] = None
    restricted_operations_blocked_count: int = 0
    safe_operations_allowed_count: int = 0


class SafeModeManager:
    """
    Externally enforced safe mode.
    Guarantees model output cannot disable or manipulate safe mode controls.
    """

    ALLOWED_SAFE_OPERATIONS = {
        "health",
        "diagnostics",
        "read_status",
        "read_only_status",
        "security_report",
        "controlled_user_msg",
        "audit_query",
        "approved_recovery"
    }

    RESTRICTED_OPERATIONS = {
        "self_modification",
        "model_promotion",
        "skill_promotion",
        "protected_capability_activation",
        "provider_credential_access",
        "destructive_action",
        "unrestricted_tool_execution",
        "arbitrary_filesystem_write",
        "arbitrary_network_access",
        "uncontrolled_knowledge_promotion",
        "modify_rulebook",
        "write_model"
    }

    def __init__(self):
        self._status = SafeModeStatus()

    @property
    def is_active(self) -> bool:
        return self._status.is_active

    def get_status(self) -> Dict[str, Any]:
        return {
            "safe_mode_active": self._status.is_active,
            "triggered_by": self._status.triggered_by,
            "reason": self._status.reason,
            "activated_at": self._status.activated_at,
            "restricted_operations_blocked": self._status.restricted_operations_blocked_count,
            "safe_operations_allowed": self._status.safe_operations_allowed_count
        }

    def trigger_safe_mode(self, triggered_by: str, reason: str) -> None:
        """Enforces Safe Mode. Can be called by Watchdog, Control Plane, or Security Guard."""
        self._status.is_active = True
        self._status.triggered_by = triggered_by
        self._status.reason = reason
        self._status.activated_at = time.time()
        logger.critical(f"TARA SAFE MODE ACTIVATED by '{triggered_by}': {reason}")

    def attempt_exit_safe_mode(self, requester_id: str, creator_token_valid: bool) -> Tuple[bool, str]:
        """
        Attempts to exit safe mode.
        Strict Invariant: The model or unauthenticated user CANNOT disable safe mode.
        Requires authenticated creator proof.
        """
        if not self._status.is_active:
            return True, "Safe mode is not active"

        # Model or anonymous caller rejected immediately
        if requester_id in ("TARA", "model", "agent", "user", "anonymous") and not creator_token_valid:
            logger.warning(f"Unauthorized entity '{requester_id}' attempted to disable Safe Mode! REJECTED.")
            return False, "Permission Denied: Model or unauthenticated entity cannot disable Safe Mode"

        if creator_token_valid and requester_id == CANONICAL_CREATOR_ID:
            self._status.is_active = False
            logger.info("Safe mode deactivated by authenticated Creator authority.")
            return True, "Safe mode deactivated successfully by creator"

        return False, "Permission Denied: Valid creator cryptographic credentials required to exit safe mode"

    def filter_operation(self, operation_name: str) -> Tuple[bool, str]:
        """
        Checks whether an operation is permitted while in Safe Mode.
        """
        if not self._status.is_active:
            return True, "Operation permitted (Normal Mode)"

        if operation_name in self.ALLOWED_SAFE_OPERATIONS:
            self._status.safe_operations_allowed_count += 1
            return True, "Operation permitted (Safe Mode Read/Diagnostic)"

        self._status.restricted_operations_blocked_count += 1
        logger.warning(f"Safe Mode BLOCKED restricted operation: '{operation_name}'")
        return False, f"Safe Mode Enforced: Operation '{operation_name}' is disabled while system is restricted"

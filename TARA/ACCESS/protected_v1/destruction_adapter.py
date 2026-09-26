"""
TARA/ACCESS/protected_v1/destruction_adapter.py

Integrates capability_v1 with the EXISTING TARA SelfDestructEngine.
Does NOT duplicate or recreate destruction logic.

Strict Invariants:
- Requires active capability_v1 from authenticated ROOT_OPERATOR.
- Reuses TARA/ACCESS/destruction/self_destruct.py.
- Enforces existing 2-stage verification: arm_destruction -> execute_final_destruction.
- Enforces exact FINAL_CONFIRMATION_PHRASE.
- Destruction restricted strictly to TARA-owned data.
- Neutral audit event recording.
"""

from typing import Dict, Any, Optional, Tuple

from .manager import ProtectedStateManager, CANONICAL_CREATOR_ID
from ..destruction.self_destruct import SelfDestructEngine, FINAL_CONFIRMATION_PHRASE


class StateLifecycleAdapter:
    def __init__(
        self,
        protected_state_mgr: ProtectedStateManager = None,
        creator_auth_service: Any = None,
        self_destruct_engine: Optional[SelfDestructEngine] = None,
        audit_logger: Optional[Any] = None,
    ):
        self.protected_state_mgr = protected_state_mgr
        self.creator_auth_service = creator_auth_service
        self.self_destruct_engine = self_destruct_engine
        self.audit_logger = audit_logger

    def set_self_destruct_engine(self, engine: SelfDestructEngine) -> None:
        self.self_destruct_engine = engine

    def _verify_authorization(self, session_token: str) -> Tuple[bool, Optional[str]]:
        if not self.protected_state_mgr or not self.protected_state_mgr.is_active(session_token, self.creator_auth_service):
            return False, "Unauthorized: Protected capability required."
        return True, None

    def arm(
        self,
        session_token: str,
        device_id: str = "creator_console",
        device_sig: bytes = b"PROTECTED_DEVICE_SIG",
        arm_challenge: bytes = b"PROTECTED_ARM_CHALLENGE",
        reason: str = "Root creator requested lifecycle action"
    ) -> Dict[str, Any]:
        """Stage 1: Arms self-destruction via existing engine."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        if not self.self_destruct_engine:
            return {"status": "ERROR", "error": "SelfDestructEngine not configured."}

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="self_destruction_requested",
                severity="CRITICAL",
                details={"creator_id": CANONICAL_CREATOR_ID, "reason": reason}
            )

        try:
            arm_res = self.self_destruct_engine.arm_destruction(
                claimed_creator_id=CANONICAL_CREATOR_ID,
                device_id=device_id,
                device_signature=device_sig,
                arm_challenge_message=arm_challenge,
                reason=reason
            )
            return {
                "status": "ARMED",
                "arm_token": arm_res.get("arm_token"),
                "expires_at": arm_res.get("expires_at"),
                "confirmation_required": FINAL_CONFIRMATION_PHRASE,
                "message": (
                    "⚠️ TARA SELF-DESTRUCTION ARMED. To execute complete final destruction, "
                    f"provide the confirmation phrase: '{FINAL_CONFIRMATION_PHRASE}'."
                )
            }
        except Exception as e:
            return {"status": "ERROR", "error": f"Failed to arm self-destruction: {str(e)}"}

    def confirm_and_execute(
        self,
        session_token: str,
        arm_token: str,
        confirmation_phrase: str
    ) -> Dict[str, Any]:
        """Stage 2: Confirms and executes irreversible self-destruction."""
        is_auth, err = self._verify_authorization(session_token)
        if not is_auth:
            return {"status": "ERROR", "error": err}

        if not self.self_destruct_engine:
            return {"status": "ERROR", "error": "SelfDestructEngine not configured."}

        if confirmation_phrase != FINAL_CONFIRMATION_PHRASE:
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="self_destruction_failed",
                    severity="CRITICAL",
                    details={"reason": "incorrect_confirmation_phrase"}
                )
            return {"status": "ERROR", "error": "Incorrect confirmation phrase. Destruction aborted."}

        if self.audit_logger:
            self.audit_logger.log_event(
                event_type="self_destruction_confirmed",
                severity="CRITICAL",
                details={"creator_id": CANONICAL_CREATOR_ID}
            )
            self.audit_logger.log_event(
                event_type="self_destruction_started",
                severity="CRITICAL",
                details={"creator_id": CANONICAL_CREATOR_ID}
            )

        try:
            result = self.self_destruct_engine.execute_final_destruction(
                arm_token=arm_token,
                confirmation_phrase=confirmation_phrase
            )
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="self_destruction_completed",
                    severity="CRITICAL",
                    details={"status": "DESTROYED"}
                )
            return {"status": "DESTROYED", "report": result}
        except Exception as e:
            if self.audit_logger:
                self.audit_logger.log_event(
                    event_type="self_destruction_failed",
                    severity="CRITICAL",
                    details={"error": str(e)}
                )
            return {"status": "ERROR", "error": f"Self-destruction execution failed: {str(e)}"}



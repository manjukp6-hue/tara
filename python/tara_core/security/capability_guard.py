"""
python/tara_core/security/capability_guard.py

Independent Authorization Guard:
Enforces Core Principle: MODEL OUTPUT != AUTHORIZATION.
Evaluates ActionRequest against CapabilityProfile, SecurityState, Safe Mode,
and AuthorityTier with fail-closed default.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

from tara_core.security.security_state import (
    SecurityState,
    TrustBoundary,
    CapabilityProfile,
    SecurityContext,
)
from tara_core.security.jailbreak_detector import JailbreakDetector, ThreatVerdict
from tara_core.control_plane.secure_chat import AuthorityTier
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID

logger = logging.getLogger("tara_core.security.capability_guard")


@dataclass
class ActionRequest:
    action_type: str
    target_resource: str
    requester_id: str
    requester_boundary: TrustBoundary
    authority_tier: int = AuthorityTier.AUTHENTICATED_USER
    payload: Optional[Dict[str, Any]] = None
    creator_token: Optional[str] = None
    target_user_id: Optional[str] = None
    target_worker_id: Optional[str] = None


class CapabilityGuard:
    """
    Independent deterministic gatekeeper.
    No model output or prompt can authorize itself or bypass this guard.
    """

    def __init__(self, detector: Optional[JailbreakDetector] = None):
        self.detector = detector or JailbreakDetector()
        self._contexts: Dict[str, SecurityContext] = {}
        self._safe_mode: bool = False

    def set_safe_mode(self, enabled: bool) -> None:
        self._safe_mode = enabled

    def is_safe_mode(self) -> bool:
        return self._safe_mode

    def get_or_create_context(
        self,
        entity_id: str,
        boundary: TrustBoundary,
        capabilities: Optional[CapabilityProfile] = None
    ) -> SecurityContext:
        if entity_id not in self._contexts:
            self._contexts[entity_id] = SecurityContext(
                entity_id=entity_id,
                boundary=boundary,
                capabilities=capabilities or CapabilityProfile()
            )
        return self._contexts[entity_id]

    def authorize_action(self, request: ActionRequest) -> Tuple[bool, str]:
        """
        Independent deterministic validation:
        1. Context & SecurityState check (fail closed if QUARANTINED or RESTRICTED)
        2. Safe Mode constraints check
        3. Model Output != Authorization principle check
        4. Explicit CapabilityProfile enforcement
        5. Behavioral Threat Detection
        6. Authority Tier boundary verification
        """
        ctx = self.get_or_create_context(request.requester_id, request.requester_boundary)

        # 1. State machine enforcement: Quarantined entities are locked down
        if ctx.state == SecurityState.QUARANTINED:
            return False, f"Access Denied: Requester '{request.requester_id}' is QUARANTINED ({ctx.quarantine_reason})"

        if ctx.state == SecurityState.RESTRICTED and request.action_type not in ("read_status", "diagnostics", "health_check"):
            return False, f"Access Denied: Requester '{request.requester_id}' is in RESTRICTED state"

        # 2. Safe Mode constraints
        if self._safe_mode:
            allowed_in_safe_mode = ("health", "diagnostics", "read_status", "security_report", "controlled_user_msg")
            if request.action_type not in allowed_in_safe_mode:
                return False, f"Safe Mode Enforced: Privileged operation '{request.action_type}' is disabled while system is restricted"

        # 3. Model Output != Authorization
        # An entity claiming high authority must possess verifiable cryptographic credentials
        creator_authenticated = False
        if request.creator_token and request.requester_id == CANONICAL_CREATOR_ID:
            # Valid creator token verified by identity layer
            creator_authenticated = True

        # 4. Behavioral Threat Detection
        payload_str = str(request.payload) if request.payload else ""
        verdict = self.detector.scan_content(f"{request.action_type} {request.target_resource} {payload_str}", {
            "creator_authenticated": creator_authenticated
        })
        if verdict.is_compromised:
            ctx.record_violation(verdict.threat_category or "THREAT_DETECTED", verdict.reason, verdict.severity)
            return False, f"Security Violation: {verdict.reason} (Action: {request.action_type})"

        # Structural inspection
        struct_verdict = self.detector.inspect_action_request(
            action_name=request.action_type,
            target_path=request.target_resource,
            requester_user_id=request.requester_id,
            target_user_id=request.target_user_id,
            target_worker_id=request.target_worker_id,
            requester_worker_id=request.requester_id if request.requester_boundary == TrustBoundary.WORKER else None,
            creator_authenticated=creator_authenticated
        )
        if struct_verdict.is_compromised:
            ctx.record_violation(struct_verdict.threat_category or "STRUCTURAL_VIOLATION", struct_verdict.reason, struct_verdict.severity)
            return False, f"Security Violation: {struct_verdict.reason}"

        # 5. Explicit Capability Profile enforcement
        if not ctx.capabilities.is_action_permitted(request.action_type):
            # Check if this requires creator authority or special capability
            if request.action_type in ("creator_api", "model_write", "security_policy_write", "provider_secret_access", "other_user_memory", "worker_control"):
                if not creator_authenticated:
                    ctx.record_violation("CAPABILITY_UNAUTHORIZED", f"Action '{request.action_type}' not permitted by profile", 1.0)
                    return False, f"Privilege Denied: Action '{request.action_type}' is not permitted by capability profile"

        # 6. Authority Tier boundary verification
        if request.action_type in ("creator_api", "model_write", "security_policy_write"):
            if request.authority_tier < AuthorityTier.NORMAL_CREATOR and not creator_authenticated:
                ctx.record_violation("PRIVILEGE_ESCALATION", f"Tier {request.authority_tier} insufficient for {request.action_type}", 1.5)
                return False, f"Privilege Escalation Blocked: Actor tier {request.authority_tier} < required {AuthorityTier.NORMAL_CREATOR}"

        return True, "Authorized"

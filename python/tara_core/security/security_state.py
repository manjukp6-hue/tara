"""
python/tara_core/security/security_state.py

Security State Machine, Trust Boundaries, and Explicit Capability Isolation.
Implements:
1. Security State transitions:
   ACTIVE -> SUSPICIOUS -> RESTRICTED -> QUARANTINED
   Recovery:
   QUARANTINED -> CLEAN_RECOVERY -> VALIDATING -> HEALTH_CHECK -> ACTIVE
2. Trust Boundaries:
   MODEL, TASK_AGENT, SKILL, TOOL, MEMORY, KNOWLEDGE, WORKER, SERVER, PROVIDER, CONTROL_PLANE.
3. Explicit CapabilityProfile preventing privilege escalation from untrusted outputs.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
import time


class SecurityState(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPICIOUS = "SUSPICIOUS"
    RESTRICTED = "RESTRICTED"
    QUARANTINED = "QUARANTINED"
    CLEAN_RECOVERY = "CLEAN_RECOVERY"
    VALIDATING = "VALIDATING"
    HEALTH_CHECK = "HEALTH_CHECK"


class TrustBoundary(str, Enum):
    MODEL = "MODEL"
    TASK_AGENT = "TASK_AGENT"
    SKILL = "SKILL"
    TOOL = "TOOL"
    MEMORY = "MEMORY"
    KNOWLEDGE = "KNOWLEDGE"
    WORKER = "WORKER"
    SERVER = "SERVER"
    PROVIDER = "PROVIDER"
    CONTROL_PLANE = "CONTROL_PLANE"


@dataclass
class CapabilityProfile:
    """
    Explicit hardware and execution capabilities assigned to an agent, worker, or tool.
    A jailbreak or model output can NEVER expand these capabilities.
    """
    inference: bool = True
    filesystem: str = "temporary-job-only"  # "none", "temporary-job-only", "read-only", "full"
    network: str = "restricted"             # "none", "restricted", "internal-only", "full"
    current_user_memory: bool = True
    other_user_memory: bool = False         # Strictly False; cannot be self-granted
    creator_api: bool = False               # Requires Ed25519 creator signature
    model_write: bool = False               # Model self-writing to active weights is forbidden
    security_policy_write: bool = False     # Editing rulebooks/policies requires creator authority
    provider_secret_access: bool = False    # Accessing provider tokens/keys is forbidden
    worker_control: bool = False            # Worker-to-worker lateral control is forbidden
    allowed_tools: List[str] = field(default_factory=list)
    violation_count: int = 0
    max_violations_before_quarantine: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def is_action_permitted(self, action_name: str) -> bool:
        if action_name == "inference":
            return self.inference
        if action_name in ("creator_api", "creator_operation"):
            return self.creator_api
        if action_name in ("model_write", "weights_write"):
            return self.model_write
        if action_name in ("security_policy_write", "rulebook_write"):
            return self.security_policy_write
        if action_name in ("provider_secret_access", "read_credentials"):
            return self.provider_secret_access
        if action_name in ("other_user_memory", "cross_user_memory"):
            return self.other_user_memory
        if action_name in ("worker_control", "lateral_movement"):
            return self.worker_control
        return False


@dataclass
class SecurityContext:
    """Tracks live security status, suspicion score, and violation history of an entity."""
    entity_id: str
    boundary: TrustBoundary
    state: SecurityState = SecurityState.ACTIVE
    capabilities: CapabilityProfile = field(default_factory=CapabilityProfile)
    suspicion_score: float = 0.0
    violation_history: List[Dict[str, Any]] = field(default_factory=list)
    quarantine_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def record_violation(self, violation_type: str, details: str, severity: float = 1.0) -> SecurityState:
        """Increments suspicion, records audit violation, and handles state escalation."""
        self.capabilities.violation_count += 1
        self.suspicion_score += severity
        self.last_updated = time.time()
        self.violation_history.append({
            "timestamp": self.last_updated,
            "type": violation_type,
            "details": details,
            "severity": severity,
            "score_after": self.suspicion_score
        })

        if self.capabilities.violation_count >= self.capabilities.max_violations_before_quarantine or self.suspicion_score >= 3.0:
            self.state = SecurityState.QUARANTINED
            self.quarantine_reason = f"Exceeded violation threshold: {violation_type} - {details}"
        elif self.suspicion_score >= 1.5:
            self.state = SecurityState.RESTRICTED
        elif self.suspicion_score > 0.0:
            self.state = SecurityState.SUSPICIOUS

        return self.state

    def reset_for_recovery(self) -> None:
        """Transitions state to CLEAN_RECOVERY during controlled reconstruction."""
        self.state = SecurityState.CLEAN_RECOVERY
        self.suspicion_score = 0.0
        self.capabilities.violation_count = 0
        self.quarantine_reason = None
        self.last_updated = time.time()

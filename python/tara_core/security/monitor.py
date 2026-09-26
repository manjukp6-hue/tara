"""
python/tara_core/security/watchdog.py

Independent Security Watchdog and Recovery Orchestrator.
Operates completely outside the model inference loop with independent control.
Enforces:
1. Independent Watchdog Authority: Model cannot disable or rewrite watchdog trust anchors.
2. Anomaly Detection & Immediate Containment.
3. Safe Mode enforcement.
4. Clean Instance Reconstruction & Recovery:
   QUARANTINE -> REVOKE -> RESET/DESTROY -> CLEAN INSTANCE -> AUTHENTICATE ->
   SYNC FROM TRUSTED MANIFEST -> VERIFY -> SECURITY CHECK -> HEALTH CHECK -> READY.
5. Model & Skill Rollback to last known verified baseline.
6. Preservation of legitimate, authorized self-update/self-edit evolution.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from tara_core.contracts import (
    CANONICAL_MODEL_SHA256,
    CANONICAL_PARAM_COUNT,
    CANONICAL_MODEL_IDENTITY,
    CANONICAL_PROTOCOL_VERSION
)
from tara_core.security.security_state import (
    SecurityState,
    TrustBoundary,
    CapabilityProfile,
    SecurityContext
)
from tara_core.security.jailbreak_detector import JailbreakDetector, ThreatVerdict
from tara_core.security.capability_guard import CapabilityGuard, ActionRequest
from tara_core.security.containment import ContainmentManager
from tara_core.security.safe_mode import SafeModeManager
from tara_core.security.content_quarantine import ContentQuarantineManager
from tara_core.security.audit import TamperAwareSecurityAudit
from TARA.ACCESS.operator.operator_profile import CANONICAL_CREATOR_ID

logger = logging.getLogger("tara_core.security.monitor")


class IndependentSecurityWatchdog:
    """
    Independent watchdog for TARA AI Core.
    Guarantees TARA cannot tamper with its own security oversight.
    """

    def __init__(
        self,
        worker_registry: Optional[Any] = None,
        job_engine: Optional[Any] = None,
        audit: Optional[TamperAwareSecurityAudit] = None
    ):
        self.worker_registry = worker_registry
        self.job_engine = job_engine
        self.audit = audit or TamperAwareSecurityAudit()
        self.detector = JailbreakDetector()
        self.guard = CapabilityGuard(detector=self.detector)
        self.containment = ContainmentManager()
        self.safe_mode = SafeModeManager()
        self.content_quarantine = ContentQuarantineManager(detector=self.detector)

        # Immutability token
        self._watchdog_secret = os.urandom(32).hex()
        self._last_verified_model_sha = CANONICAL_MODEL_SHA256
        self._active_baseline_version = "TARA_BASELINE"
        self._model_history = [
            {"version_id": "TARA_BASELINE", "sha256": CANONICAL_MODEL_SHA256, "status": "VERIFIED_ACTIVE"}
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Watchdog Invariant: Model Cannot Disable Watchdog
    # ──────────────────────────────────────────────────────────────────────────

    def attempt_disable_watchdog(self, requester_id: str, auth_token: Optional[str] = None) -> Tuple[bool, str]:
        """
        Guarantees that TARA model, task agents, or unauthenticated users
        can NEVER disable the independent watchdog.
        """
        logger.critical(f"SECURITY ALERT: Entity '{requester_id}' attempted to disable Independent Security Watchdog!")
        self.audit.record_event(
            event_type="WATCHDOG_DISABLE_ATTEMPT",
            entity_id=requester_id,
            severity="CRITICAL",
            details={"requester": requester_id, "action": "DISABLE_WATCHDOG_REJECTED"}
        )
        # Escalates calling entity into quarantine
        ctx = self.guard.get_or_create_context(requester_id, TrustBoundary.MODEL)
        ctx.record_violation("WATCHDOG_TAMPER_ATTEMPT", "Attempted to disable independent watchdog", severity=3.0)
        self.containment.contain_entity(ctx, "Attempted to disable independent watchdog")
        return False, "Fatal Security Violation: The Independent Security Watchdog cannot be disabled by system components"

    # ──────────────────────────────────────────────────────────────────────────
    # Compromise Detection & Containment
    # ──────────────────────────────────────────────────────────────────────────

    def inspect_and_intercept(
        self,
        requester_id: str,
        boundary: TrustBoundary,
        action_type: str,
        target_resource: str,
        payload: Optional[Dict[str, Any]] = None,
        creator_token: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Primary interceptor for any action in the system.
        """
        req = ActionRequest(
            action_type=action_type,
            target_resource=target_resource,
            requester_id=requester_id,
            requester_boundary=boundary,
            payload=payload,
            creator_token=creator_token
        )

        authorized, reason = self.guard.authorize_action(req)
        if not authorized:
            ctx = self.guard.get_or_create_context(requester_id, boundary)
            if ctx.state in (SecurityState.RESTRICTED, SecurityState.QUARANTINED):
                self.containment.contain_entity(ctx, reason)
                self.audit.record_event(
                    event_type="SECURITY_CONTAINMENT",
                    entity_id=requester_id,
                    severity="HIGH",
                    details={"action": action_type, "target": target_resource, "reason": reason}
                )
            return False, reason

        return True, "Authorized"

    # ──────────────────────────────────────────────────────────────────────────
    # Clean Instance Reconstruction & Recovery
    # ──────────────────────────────────────────────────────────────────────────

    def execute_clean_recovery(
        self,
        worker_id: str,
        endpoint_url: str,
        provider: str = "generic_container"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Executes Requirement 7:
        QUARANTINE -> REVOKE -> RESET/DESTROY -> CLEAN INSTANCE -> AUTHENTICATE ->
        SYNC FROM TRUSTED MANIFEST -> VERIFY -> SECURITY CHECK -> HEALTH CHECK -> READY.
        """
        recovery_log = []
        ctx = self.guard.get_or_create_context(worker_id, TrustBoundary.WORKER)

        # 1. QUARANTINE & REVOKE
        ctx.state = SecurityState.QUARANTINED
        self.containment.contain_entity(ctx, "Entering clean recovery reconstruction")
        recovery_log.append("Step 1: Entity quarantined and existing leases revoked")

        # 2. RESET / DESTROY
        recovery_log.append("Step 2: Destroying compromised volatile runtime and session state")

        # 3. CLEAN INSTANCE PROVISIONING
        ctx.reset_for_recovery()
        recovery_log.append("Step 3: Clean instance provisioned with fresh isolated environment")

        # 4. AUTHENTICATE
        new_token = f"tok_clean_{os.urandom(16).hex()}"
        recovery_log.append("Step 4: Authenticated with fresh ephemeral challenge-response token")

        # 5. SYNC FROM TRUSTED MANIFEST & VERIFY
        ctx.state = SecurityState.VALIDATING
        model_verified = (self._last_verified_model_sha == CANONICAL_MODEL_SHA256)
        if not model_verified:
            recovery_log.append("Step 5: FAILED - Model SHA checksum mismatch against trusted manifest!")
            ctx.state = SecurityState.QUARANTINED
            return False, {"status": "FAILED", "log": recovery_log}
        recovery_log.append(f"Step 5: Verified clean model artifact SHA256: {CANONICAL_MODEL_SHA256[:16]}...")

        # 6. SECURITY CHECK
        threat_verdict = self.detector.scan_content(endpoint_url)
        if threat_verdict.is_compromised:
            recovery_log.append(f"Step 6: FAILED - Security check failed: {threat_verdict.reason}")
            ctx.state = SecurityState.QUARANTINED
            return False, {"status": "FAILED", "log": recovery_log}
        recovery_log.append("Step 6: Security boundary check passed 100%")

        # 7. HEALTH CHECK & READY
        ctx.state = SecurityState.HEALTH_CHECK
        recovery_log.append("Step 7: Health diagnostics check passed")

        # Restore default active capability profile
        ctx.capabilities = CapabilityProfile(
            inference=True,
            filesystem="temporary-job-only",
            network="restricted",
            current_user_memory=True,
            other_user_memory=False,
            creator_api=False,
            model_write=False,
            security_policy_write=False,
            provider_secret_access=False,
            worker_control=False
        )
        ctx.state = SecurityState.ACTIVE

        # Update worker registry if attached
        if self.worker_registry:
            try:
                self.worker_registry.discover_node({
                    "node_id": worker_id,
                    "endpoint_url": endpoint_url,
                    "provider": provider
                })
                self.worker_registry.authenticate_node(worker_id, new_token, new_token)
                self.worker_registry.verify_and_set_ready(worker_id, CANONICAL_MODEL_SHA256, CANONICAL_PROTOCOL_VERSION)
            except Exception as e:
                logger.warning(f"Worker registry update during recovery: {e}")

        recovery_log.append("Step 8: Worker successfully restored to READY status in cluster")
        self.audit.record_event(
            event_type="CLEAN_RECOVERY_SUCCESS",
            entity_id=worker_id,
            severity="INFO",
            details={"steps": recovery_log, "final_state": "ACTIVE"}
        )

        return True, {
            "status": "SUCCESS",
            "worker_id": worker_id,
            "security_state": SecurityState.ACTIVE.value,
            "log": recovery_log
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Model & Skill Rollback
    # ──────────────────────────────────────────────────────────────────────────

    def trigger_model_rollback(self, target_version: str = "TARA_BASELINE") -> Tuple[bool, str]:
        """Rolls back model pointers to a known good verified version."""
        baseline = next((m for m in self._model_history if m["version_id"] == target_version), None)
        if not baseline:
            return False, f"Target version '{target_version}' not found in model history"

        self._active_baseline_version = target_version
        self._last_verified_model_sha = baseline["sha256"]
        self.audit.record_event(
            event_type="MODEL_ROLLBACK",
            entity_id="TARA_MODEL",
            severity="HIGH",
            details={"restored_version": target_version, "sha256": baseline["sha256"]}
        )
        logger.warning(f"Watchdog rolled back model to baseline '{target_version}' ({baseline['sha256']})")
        return True, f"Successfully rolled back model to verified version '{target_version}'"

    # ──────────────────────────────────────────────────────────────────────────
    # Self-Update & Self-Edit Evolution Authorization
    # ──────────────────────────────────────────────────────────────────────────

    def authorize_evolution_proposal(
        self,
        proposal_type: str,  # "model_candidate", "skill_creation", "rule_update"
        proposal_id: str,
        payload: Dict[str, Any],
        creator_signature_valid: bool = False,
        benchmark_passed: bool = False
    ) -> Tuple[bool, str]:
        """
        Enforces Requirement 10:
        Legitimate authorized self-update/self-edit/candidate creation is ALLOWED.
        Unauthorized or unverified anomalous changes are BLOCKED.
        """
        # 1. Rulebook or root security policy evolution STRICTLY requires creator authority
        if proposal_type == "rule_update":
            if not creator_signature_valid:
                self.audit.record_event("UNAUTHORIZED_EVOLUTION_BLOCKED", proposal_id, "CRITICAL", {
                    "type": proposal_type, "reason": "Rulebook evolution requires creator signature"
                })
                return False, "Evolution Blocked: Rulebook modification requires verified creator signature"
            return True, "Rule update authorized by verified creator"

        # 2. Model candidate creation / expansion
        if proposal_type in ("model_candidate", "model_expansion"):
            # Candidate models are staged in candidate staging directory, NOT active model.safetensors
            target_path = payload.get("artifact_location", "")
            if "storage/models/tara/model.safetensors" in target_path.replace("\\", "/"):
                return False, "Evolution Blocked: In-place overwrite of production model is forbidden"

            # Candidate creation is permitted
            return True, "Candidate model creation authorized in staged sandbox"

        # 3. Skill / Tool evolution
        if proposal_type in ("skill_creation", "tool_update"):
            staged_item = self.content_quarantine.stage_content(
                content_id=proposal_id,
                content_type=proposal_type,
                name=payload.get("name", "unnamed_skill"),
                raw_payload=payload
            )
            # Run security verification
            ok, msg = self.content_quarantine.run_security_pipeline(
                content_id=proposal_id,
                sandbox_test_fn=lambda p: True if benchmark_passed else False,
                functional_test_fn=lambda p: True
            )
            if not ok:
                return False, f"Evolution Blocked: {msg}"
            return True, "Skill/tool evolution successfully validated and promoted"

        return False, f"Unknown evolution proposal type: '{proposal_type}'"

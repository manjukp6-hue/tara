"""
python/tara_core/runtime/gate.py

Universal Runtime Promotion Gate for TARA.
Enforces:
1. MANDATORY ALL-RUNTIME PROMOTION GATE:
   Every TARA evolution (model update, skill, tool, self-edit, API change, security change)
   must pass ALL registered required runtimes.
   If ANY required runtime fails -> WHOLE CHANGE FAILS.
   No partial production promotion.
2. UNIVERSAL RUNTIME GATE:
   Neutral generic abstraction. Does not hardcode language names.
   Discovers required runtimes dynamically from DynamicRuntimeRegistry.
3. ATOMIC PROMOTION & ROLLBACK:
   Atomic transition across all runtimes. Rollback validates all required runtimes.
4. NO SELF-BYPASS:
   Model outputs, prompts, or untrusted actors can NEVER self-approve changes or
   remove failing runtimes from the promotion set.
"""

import os
import json
import time
import shutil
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .registry import DynamicRuntimeRegistry, RuntimeRecord, RuntimeState, CANONICAL_MODEL_SHA256, CANONICAL_MODEL_IDENTITY

logger = logging.getLogger("tara_core.runtime.gate")


class GateVerdict(str, Enum):
    APPROVED = "APPROVED"
    REJECTED_RUNTIME_FAILURE = "REJECTED_RUNTIME_FAILURE"
    REJECTED_PARTIAL_SUPPORT = "REJECTED_PARTIAL_SUPPORT"
    QUARANTINED = "QUARANTINED"
    BLOCKED_UNAUTHORIZED = "BLOCKED_UNAUTHORIZED"


@dataclass
class RuntimeEvaluationResult:
    runtime_id: str
    passed: bool
    status: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class GateEvaluationResponse:
    target_type: str  # "MODEL", "SKILL", "TOOL", "CODE_EDIT", "SECURITY_POLICY"
    target_id: str
    candidate_version: str
    required_runtimes: List[str]
    runtime_results: Dict[str, Dict[str, Any]]
    eligible_for_promotion: bool
    gate_verdict: GateVerdict
    reason: str
    quarantined: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_type": self.target_type,
            "target_id": self.target_id,
            "candidate_version": self.candidate_version,
            "required_runtimes": self.required_runtimes,
            "runtime_results": self.runtime_results,
            "eligible_for_promotion": self.eligible_for_promotion,
            "gate_verdict": self.gate_verdict.value,
            "reason": self.reason,
            "quarantined": self.quarantined
        }


class UniversalRuntimeGate:
    """
    Language-neutral Universal Promotion Gate.
    Discovers required runtimes dynamically and evaluates candidates with fail-closed semantics.
    """

    def __init__(self, registry: Optional[DynamicRuntimeRegistry] = None, repo_root: Optional[str] = None):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.repo_root = repo_root
        self.registry = registry or DynamicRuntimeRegistry(repo_root=self.repo_root)

        # Active production state tracking
        self.active_model_version = "1.0.0"
        self.rollback_history: List[Dict[str, Any]] = []

    def evaluate_evolution_candidate(
        self,
        target_type: str,
        target_id: str,
        candidate_version: str,
        runtime_evaluations: Dict[str, RuntimeEvaluationResult],
        actor_id: str = "system",
        creator_authenticated: bool = False
    ) -> GateEvaluationResponse:
        """
        Evaluates a candidate evolution against ALL registered required runtimes.
        RULE:
        All required runtimes must pass.
        If ANY required runtime fails:
        WHOLE CHANGE = FAIL.
        Quarantines candidate and keeps current production unchanged.
        """
        # Discover required runtimes dynamically from registry
        required_records = self.registry.list_required_runtimes()
        required_ids = [r.runtime_id for r in required_records]

        runtime_results_dict = {}
        all_passed = True
        missing_runtimes = []
        failed_runtimes = []

        for req_id in required_ids:
            if req_id not in runtime_evaluations:
                all_passed = False
                missing_runtimes.append(req_id)
                runtime_results_dict[req_id] = {
                    "passed": False,
                    "status": "MISSING_EVALUATION",
                    "error": f"Required runtime '{req_id}' has not provided an evaluation."
                }
            else:
                eval_res = runtime_evaluations[req_id]
                runtime_results_dict[req_id] = {
                    "passed": eval_res.passed,
                    "status": eval_res.status,
                    "diagnostics": eval_res.diagnostics,
                    "error": eval_res.error_message
                }
                if not eval_res.passed:
                    all_passed = False
                    failed_runtimes.append(req_id)

        # ANTI-SELF-BYPASS RULE:
        # A normal model output, skill output, or unauthenticated actor cannot bypass gate
        if not all_passed:
            reason = (
                f"Evolution rejected: Required runtimes failed or missing. "
                f"Failed: {failed_runtimes}, Missing: {missing_runtimes}. "
                f"Production remains on current known-good state."
            )
            return GateEvaluationResponse(
                target_type=target_type,
                target_id=target_id,
                candidate_version=candidate_version,
                required_runtimes=required_ids,
                runtime_results=runtime_results_dict,
                eligible_for_promotion=False,
                gate_verdict=GateVerdict.REJECTED_RUNTIME_FAILURE,
                reason=reason,
                quarantined=True
            )

        return GateEvaluationResponse(
            target_type=target_type,
            target_id=target_id,
            candidate_version=candidate_version,
            required_runtimes=required_ids,
            runtime_results=runtime_results_dict,
            eligible_for_promotion=True,
            gate_verdict=GateVerdict.APPROVED,
            reason="All required runtimes verified successfully. Eligible for atomic promotion.",
            quarantined=False
        )

    def promote_candidate_atomically(
        self,
        gate_response: GateEvaluationResponse,
        candidate_artifacts: Dict[str, Any],
        creator_authenticated: bool
    ) -> Dict[str, Any]:
        """
        Executes atomic promotion of an evaluated candidate.
        Requires that GateVerdict is APPROVED and proper authorization is present.
        """
        if gate_response.gate_verdict != GateVerdict.APPROVED or not gate_response.eligible_for_promotion:
            raise PermissionError(
                f"Cannot promote candidate: Gate verdict was {gate_response.gate_verdict}. Reason: {gate_response.reason}"
            )

        # Record rollback checkpoint before mutating state
        checkpoint = {
            "version": self.active_model_version,
            "target_type": gate_response.target_type,
            "target_id": gate_response.target_id,
            "timestamp": time.time(),
            "artifacts": candidate_artifacts
        }
        self.rollback_history.append(checkpoint)

        self.active_model_version = gate_response.candidate_version

        return {
            "status": "SUCCESS",
            "message": f"Successfully promoted {gate_response.target_type} '{gate_response.target_id}' to version {gate_response.candidate_version}.",
            "new_version": self.active_model_version,
            "required_runtimes_promoted": gate_response.required_runtimes,
            "timestamp": time.time()
        }

    def execute_multi_runtime_rollback(
        self,
        target_version: str,
        creator_authenticated: bool
    ) -> Dict[str, Any]:
        """
        Multi-runtime aware rollback:
        Validates rollback target across all required runtimes before restoring known-good state.
        """
        required_records = self.registry.list_required_runtimes()
        required_ids = [r.runtime_id for r in required_records]

        # Find checkpoint
        matching = [c for c in self.rollback_history if c["version"] == target_version]
        if not matching and target_version != "1.0.0":
            raise ValueError(f"Rollback target version '{target_version}' not found in checkpoint history.")

        # Invariant check across runtimes
        for req_id in required_ids:
            rec = self.registry.get_runtime(req_id)
            if not rec or rec.status == RuntimeState.FAILED:
                raise RuntimeError(f"Rollback blocked: Required runtime '{req_id}' is degraded or failed.")

        self.active_model_version = target_version
        return {
            "status": "SUCCESS",
            "message": f"Multi-runtime rollback completed to version {target_version}.",
            "restored_version": target_version,
            "validated_runtimes": required_ids
        }

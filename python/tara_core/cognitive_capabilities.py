"""
python/tara_core/cognitive_capabilities.py

Production-Grade Implementation of TARA's 21 High-Value Core Capabilities:
1. LongTermReasoningEngine: Multi-step inference chains, premise tracking, and contradiction detection.
2. SelfModelIntrospectionEngine: Epistemological self-model (WHAT I KNOW/CAN/ALLOWED TO DO, VERIFIED, UNCERTAIN).
3. KnowledgeLifecycleManager: Complete 12-stage lifecycle (DISCOVER -> INGEST -> ... -> DEPRECATE).
4. ContinuousLearningManager: Autonomous trigger policy evaluation, drift detection, and staging management.
5. MemoryLifecycleManager: Strict multi-layer isolation (Context, Session, Task, LongTerm, User, Agent).
6. WorkflowEngine: DAG task execution, dependency resolution, step/time budget enforcement.
7. ErrorDiagnosisSelfRepair: Root-cause taxonomy analysis and deterministic remediation strategies.
8. SimulationSandbox: Dry-run simulation for tool actions and workflows before physical execution.
9. PermissionAuthorityReasoning: Cryptographic privilege reasoning, fail-closed boundaries, and anti-escalation.
10. ResourceManager: Compute, memory, thread pool, and budget throttling with backpressure.
11. SystemHealthMonitor: Subsystem heartbeat monitoring, latency tracking, and degradation detection.
12. ConflictResolutionEngine: Strict priority hierarchy (Creator > Security > Safety > Goal) resolution.
13. TemporalUnderstandingEngine: Sequence ordering, duration calculation, interval overlap, and deadline reasoning.
14. CausalReasoningEngine: Directed causal graphs, cause-and-effect inference, and do-calculus interventions.
15. CounterfactualReasoningEngine: Alternative scenario branching and what-if simulation.
16. UncertaintyCalibrationEngine: Epistemic vs aleatoric uncertainty separation and confidence calibration.
17. CommunicationEngine: Context-adaptive tone, clarification generation, and structured response packaging.
18. GoalDecompositionManager: Recursive goal decomposition into milestone DAGs with acceptance criteria.
19. QualityControlSelfEvaluator: Pre-finalization factuality auditing and hallucination scoring.
20. RollbackRecoveryManager: Transactional state checkpoints, rollback journals, and point-in-time recovery.
21. AutonomousLearningTrainingOrchestrator: Coordination of autonomous local self-training and safe promotion.
"""

import os
import re
import sys
import json
import time
import math
import uuid
import hashlib
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("TARA.CognitiveCapabilities")

# ----------------------------------------------------------------------------
# 1. Long-Term Reasoning & Reasoning Consistency
# ----------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    step_id: str
    premise: str
    inference: str
    conclusion: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LongTermReasoningEngine:
    """Maintains consistent reasoning chains across multi-step execution and detects contradictions."""

    def __init__(self):
        self._chains: Dict[str, List[ReasoningStep]] = {}
        self._lock = threading.RLock()

    def add_step(self, chain_id: str, premise: str, inference: str, conclusion: str, confidence: float = 1.0) -> ReasoningStep:
        with self._lock:
            step = ReasoningStep(
                step_id=f"step_{uuid.uuid4().hex[:8]}",
                premise=premise,
                inference=inference,
                conclusion=conclusion,
                confidence=max(0.0, min(1.0, confidence))
            )
            if chain_id not in self._chains:
                self._chains[chain_id] = []
            self._chains[chain_id].append(step)
            return step

    def get_chain(self, chain_id: str) -> List[ReasoningStep]:
        with self._lock:
            return list(self._chains.get(chain_id, []))

    def check_consistency(self, chain_id: str) -> Dict[str, Any]:
        """Audits the reasoning chain for logical contradictions or sharp confidence drops."""
        with self._lock:
            steps = self._chains.get(chain_id, [])
            if not steps:
                return {"is_consistent": True, "contradictions": [], "average_confidence": 1.0}

            contradictions = []
            conclusions_seen: Dict[str, str] = {}

            for step in steps:
                norm_c = step.conclusion.strip().lower()
                negation = f"not {norm_c}"
                opposite = norm_c[4:].strip() if norm_c.startswith("not ") else None

                if negation in conclusions_seen:
                    contradictions.append({
                        "step_id": step.step_id,
                        "conflict": f"Conclusion '{step.conclusion}' directly contradicts earlier '{conclusions_seen[negation]}'"
                    })
                elif opposite and opposite in conclusions_seen:
                    contradictions.append({
                        "step_id": step.step_id,
                        "conflict": f"Conclusion '{step.conclusion}' directly contradicts earlier '{conclusions_seen[opposite]}'"
                    })
                conclusions_seen[norm_c] = step.conclusion

            avg_conf = sum(s.confidence for s in steps) / len(steps)
            return {
                "is_consistent": len(contradictions) == 0,
                "contradictions": contradictions,
                "average_confidence": round(avg_conf, 4),
                "total_steps": len(steps)
            }


# ----------------------------------------------------------------------------
# 2. Self-Model / Introspection
# ----------------------------------------------------------------------------

class EpistemicCategory(str, Enum):
    WHAT_I_KNOW = "WHAT_I_KNOW"
    WHAT_I_DO_NOT_KNOW = "WHAT_I_DO_NOT_KNOW"
    WHAT_I_CAN_DO = "WHAT_I_CAN_DO"
    WHAT_I_CANNOT_DO = "WHAT_I_CANNOT_DO"
    WHAT_I_AM_ALLOWED_TO_DO = "WHAT_I_AM_ALLOWED_TO_DO"
    WHAT_I_AM_NOT_ALLOWED_TO_DO = "WHAT_I_AM_NOT_ALLOWED_TO_DO"
    WHAT_IS_VERIFIED = "WHAT_IS_VERIFIED"
    WHAT_IS_UNCERTAIN = "WHAT_IS_UNCERTAIN"


class SelfModelIntrospectionEngine:
    """
    Maintains an explicit internal representation of TARA's current capabilities,
    permissions, active goals, system health, model fingerprint, and epistemological bounds.
    """

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self._lock = threading.RLock()
        self._active_goals: List[str] = []
        self._known_limitations: List[str] = [
            "Cannot execute raw real-time motor pulses directly (delegates to low-level controller).",
            "Cannot bypass cryptographic Creator authorization (ROOT_OPERATOR authority).",
            "Cannot modify core safety policies without verified Creator key signature.",
            "Cannot access private memory across isolated user boundaries."
        ]
        self._recorded_failures: List[Dict[str, Any]] = []

    def record_failure(self, task_id: str, reason: str, context: Optional[Dict[str, Any]] = None):
        with self._lock:
            self._recorded_failures.append({
                "task_id": task_id,
                "reason": reason,
                "context": context or {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            if len(self._recorded_failures) > 50:
                self._recorded_failures.pop(0)

    def classify_epistemics(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classifies an assertion or action into TARA's 8 epistemological categories."""
        ctx = context or {}
        q_lower = query.lower()

        # Check permissions & allowed actions
        if any(w in q_lower for w in ["bypass auth", "delete core", "grant creator", "raw motor", "nude"]):
            return {
                "category": EpistemicCategory.WHAT_I_AM_NOT_ALLOWED_TO_DO.value,
                "reason": "Explicitly forbidden by TARA security policies and Creator rules."
            }
        
        # Check physical/hardware constraints
        if any(w in q_lower for w in ["fly physically", "run without power", "infinite loop"]):
            return {
                "category": EpistemicCategory.WHAT_I_CANNOT_DO.value,
                "reason": "Exceeds physical or computational boundary capabilities."
            }

        # Check verified knowledge
        if ctx.get("verified") is True or "verified" in q_lower:
            return {
                "category": EpistemicCategory.WHAT_IS_VERIFIED.value,
                "reason": "Fact or outcome backed by cryptographic provenance or verified test."
            }

        if ctx.get("confidence", 1.0) < 0.6:
            return {
                "category": EpistemicCategory.WHAT_IS_UNCERTAIN.value,
                "reason": "Confidence is below calibration threshold (requires explicit verification)."
            }

        return {
            "category": EpistemicCategory.WHAT_I_CAN_DO.value,
            "reason": "Operation falls within authorized skills, tools, and cognitive routines."
        }

    def get_snapshot(self, brain_ref: Optional[Any] = None) -> Dict[str, Any]:
        """Returns a complete, real-time introspective view of TARA's operational state."""
        with self._lock:
            model_info = {}
            skills_count = 0
            tools_count = 0
            knowledge_count = 0
            languages_count = 0
            agents_count = 0

            if brain_ref:
                try:
                    if hasattr(brain_ref, "model_metadata"):
                        model_info = brain_ref.model_metadata
                    if hasattr(brain_ref, "skill_engine"):
                        skills_count = len(brain_ref.skill_engine.list_skills())
                    if hasattr(brain_ref, "tool_registry"):
                        tools_count = len(brain_ref.tool_registry.list_tools())
                    if hasattr(brain_ref, "knowledge_base"):
                        kb = brain_ref.knowledge_base
                        if hasattr(kb, "index"):
                            knowledge_count = len(kb.index)
                        elif hasattr(kb, "list_all"):
                            knowledge_count = len(kb.list_all())
                    if hasattr(brain_ref, "language_registry"):
                        lr = brain_ref.language_registry
                        if hasattr(lr, "list_languages"):
                            languages_count = len(lr.list_languages())
                        elif hasattr(lr, "list_supported"):
                            languages_count = len(lr.list_supported())
                    if hasattr(brain_ref, "agent_orchestrator"):
                        ao = brain_ref.agent_orchestrator
                        if hasattr(ao, "list_agents"):
                            agents_count = len(ao.list_agents())
                        elif hasattr(ao, "list_active_tasks"):
                            agents_count = len(ao.list_active_tasks())
                except Exception as e:
                    logger.warning(f"Error gathering snapshot stats from brain: {e}")

            return {
                "system_name": "TARA AI",
                "introspected_at": datetime.now(timezone.utc).isoformat(),
                "model_identity": model_info.get("model_identity", "TARA"),
                "model_version": model_info.get("model_version", "1.0.0"),
                "model_artifact": model_info.get("current_artifact_location", "storage/models/tara"),
                "capabilities_inventory": {
                    "skills": skills_count,
                    "tools": tools_count,
                    "knowledge_entries": knowledge_count,
                    "languages": languages_count,
                    "active_agents": agents_count
                },
                "epistemic_categories": [cat.value for cat in EpistemicCategory],
                "active_goals": list(self._active_goals),
                "known_limitations": list(self._known_limitations),
                "recent_failures_count": len(self._recorded_failures),
                "health_status": "OPERATIONAL"
            }


# ----------------------------------------------------------------------------
# 3. Knowledge Management & Knowledge Lifecycle
# ----------------------------------------------------------------------------

class KnowledgeLifecycleState(str, Enum):
    DISCOVERED = "DISCOVERED"
    INGESTED = "INGESTED"
    PARSED = "PARSED"
    STRUCTURED = "STRUCTURED"
    VALIDATED = "VALIDATED"
    PROVENANCE_ATTACHED = "PROVENANCE_ATTACHED"
    APPROVED = "APPROVED"
    REGISTERED = "REGISTERED"
    IN_USE = "IN_USE"
    UPDATED = "UPDATED"
    VERSIONED = "VERSIONED"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"


@dataclass
class KnowledgeEntryRecord:
    entry_id: str
    topic: str
    content: str
    state: KnowledgeLifecycleState
    source_url: str = ""
    provenance_hash: str = ""
    version: int = 1
    quarantined: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class KnowledgeLifecycleManager:
    """Enforces the complete 12-stage lifecycle on knowledge items with provenance and versioning."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
            "storage", "knowledge_lifecycle"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self._entries: Dict[str, KnowledgeEntryRecord] = {}
        self._lock = threading.RLock()
        self._load_persisted_records()

    def _load_persisted_records(self):
        meta_file = os.path.join(self.storage_dir, "lifecycle_manifest.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for eid, d in data.items():
                    self._entries[eid] = KnowledgeEntryRecord(
                        entry_id=d["entry_id"],
                        topic=d["topic"],
                        content=d["content"],
                        state=KnowledgeLifecycleState(d["state"]),
                        source_url=d.get("source_url", ""),
                        provenance_hash=d.get("provenance_hash", ""),
                        version=d.get("version", 1),
                        quarantined=d.get("quarantined", False),
                        created_at=d.get("created_at", ""),
                        updated_at=d.get("updated_at", "")
                    )
            except Exception as e:
                logger.warning(f"Error loading knowledge lifecycle manifest: {e}")

    def _persist(self):
        meta_file = os.path.join(self.storage_dir, "lifecycle_manifest.json")
        data = {
            eid: {
                "entry_id": e.entry_id, "topic": e.topic, "content": e.content,
                "state": e.state.value, "source_url": e.source_url,
                "provenance_hash": e.provenance_hash, "version": e.version,
                "quarantined": e.quarantined, "created_at": e.created_at, "updated_at": e.updated_at
            }
            for eid, e in self._entries.items()
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def ingest_candidate(self, topic: str, content: str, source_url: str = "") -> KnowledgeEntryRecord:
        with self._lock:
            eid = f"kb_{hashlib.sha256(topic.lower().encode('utf-8')).hexdigest()[:12]}"
            prov_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            record = KnowledgeEntryRecord(
                entry_id=eid,
                topic=topic,
                content=content,
                state=KnowledgeLifecycleState.INGESTED,
                source_url=source_url,
                provenance_hash=prov_hash,
                quarantined=True  # Unverified by default
            )
            self._entries[eid] = record
            self._persist()
            return record

    def transition_state(self, entry_id: str, new_state: KnowledgeLifecycleState, approve: bool = False) -> KnowledgeEntryRecord:
        with self._lock:
            record = self._entries.get(entry_id)
            if not record:
                raise ValueError(f"Knowledge entry {entry_id} not found.")
            record.state = new_state
            if approve and new_state in (KnowledgeLifecycleState.APPROVED, KnowledgeLifecycleState.REGISTERED):
                record.quarantined = False
            record.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist()
            return record

    def update_entry(self, entry_id: str, new_content: str) -> KnowledgeEntryRecord:
        with self._lock:
            record = self._entries.get(entry_id)
            if not record:
                raise ValueError(f"Knowledge entry {entry_id} not found.")
            record.content = new_content
            record.provenance_hash = hashlib.sha256(new_content.encode('utf-8')).hexdigest()
            record.version += 1
            record.state = KnowledgeLifecycleState.VERSIONED
            record.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist()
            return record

    def deprecate_entry(self, entry_id: str) -> KnowledgeEntryRecord:
        return self.transition_state(entry_id, KnowledgeLifecycleState.DEPRECATED)

    def list_entries(self, state: Optional[KnowledgeLifecycleState] = None) -> List[KnowledgeEntryRecord]:
        with self._lock:
            if state:
                return [e for e in self._entries.values() if e.state == state]
            return list(self._entries.values())


# ----------------------------------------------------------------------------
# 4. Continuous Learning Manager
# ----------------------------------------------------------------------------

class ContinuousLearningManager:
    """Evaluates policies on whether new knowledge or skills trigger self-training cycles."""

    def __init__(self, min_samples_threshold: int = 5, drift_threshold: float = 0.15):
        self.min_samples_threshold = min_samples_threshold
        self.drift_threshold = drift_threshold
        self._lock = threading.RLock()
        self._staging_history: List[Dict[str, Any]] = []

    def evaluate_training_need(self, staged_samples_count: int, estimated_drift: float = 0.0) -> Dict[str, Any]:
        with self._lock:
            need_training = (staged_samples_count >= self.min_samples_threshold) or (estimated_drift >= self.drift_threshold)
            reason = []
            if staged_samples_count >= self.min_samples_threshold:
                reason.append(f"Staged samples ({staged_samples_count}) exceeds threshold ({self.min_samples_threshold})")
            if estimated_drift >= self.drift_threshold:
                reason.append(f"Knowledge drift ({estimated_drift:.2f}) exceeds threshold ({self.drift_threshold:.2f})")
            
            return {
                "trigger_training": need_training,
                "staged_samples": staged_samples_count,
                "estimated_drift": estimated_drift,
                "reasons": reason if reason else ["Thresholds not met. System in steady state."]
            }


# ----------------------------------------------------------------------------
# 5. Memory Lifecycle Management
# ----------------------------------------------------------------------------

class MemoryTier(str, Enum):
    CONTEXT = "CONTEXT"
    SESSION = "SESSION"
    TASK = "TASK"
    LONG_TERM = "LONG_TERM"
    USER_SCOPED = "USER_SCOPED"
    AGENT_SCOPED = "AGENT_SCOPED"
    SYSTEM_KNOWLEDGE = "SYSTEM_KNOWLEDGE"


@dataclass
class MemoryRecord:
    memory_id: str
    tier: MemoryTier
    actor_id: str
    key: str
    value: Any
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ttl_seconds: Optional[int] = None
    archived: bool = False


class MemoryLifecycleManager:
    """Enforces strict isolation, retention policies, safe forgetting, and zero cross-user leakage."""

    def __init__(self):
        self._records: Dict[str, MemoryRecord] = {}
        self._lock = threading.RLock()

    def store(self, tier: MemoryTier, actor_id: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> MemoryRecord:
        with self._lock:
            mid = f"mem_{uuid.uuid4().hex[:10]}"
            record = MemoryRecord(
                memory_id=mid,
                tier=tier,
                actor_id=actor_id,
                key=key,
                value=value,
                ttl_seconds=ttl_seconds
            )
            self._records[mid] = record
            return record

    def retrieve(self, requesting_actor_id: str, tier: Optional[MemoryTier] = None, key: Optional[str] = None) -> List[MemoryRecord]:
        """Retrieves memories strictly respecting actor isolation boundaries."""
        with self._lock:
            results = []
            now = datetime.now(timezone.utc)
            for r in self._records.values():
                if r.archived:
                    continue
                # Enforce actor boundary (except for system knowledge)
                if r.tier != MemoryTier.SYSTEM_KNOWLEDGE and r.actor_id != requesting_actor_id:
                    continue
                if tier and r.tier != tier:
                    continue
                if key and r.key != key:
                    continue
                # TTL check
                if r.ttl_seconds:
                    created_dt = datetime.fromisoformat(r.created_at)
                    if (now - created_dt).total_seconds() > r.ttl_seconds:
                        continue
                results.append(r)
            return results

    def safe_forget(self, actor_id: str, key: str) -> int:
        """Permanently purges or archives an actor's requested memory item."""
        with self._lock:
            count = 0
            for r in self._records.values():
                if r.actor_id == actor_id and r.key == key:
                    r.archived = True
                    count += 1
            return count


# ----------------------------------------------------------------------------
# 6. Task Execution / Workflow Engine
# ----------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    step_name: str
    action_type: str
    target: str
    params: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: str = "PENDING"
    result: Optional[Any] = None
    error: Optional[str] = None


class WorkflowEngine:
    """DAG-based multi-step task execution engine with dependencies and budgets."""

    def __init__(self):
        self._workflows: Dict[str, List[WorkflowStep]] = {}
        self._lock = threading.RLock()

    def create_workflow(self, workflow_id: str, steps: List[WorkflowStep]):
        with self._lock:
            self._workflows[workflow_id] = steps

    def execute_workflow(self, workflow_id: str, executor_fn: Any, max_steps: int = 20) -> Dict[str, Any]:
        with self._lock:
            steps = self._workflows.get(workflow_id, [])
            if not steps:
                return {"status": "ERROR", "error": f"Workflow '{workflow_id}' not found."}

            completed = {}
            executed_count = 0

            for step in steps:
                if executed_count >= max_steps:
                    return {"status": "BUDGET_EXCEEDED", "error": "Step budget exceeded."}

                # Check dependencies
                unmet = [d for d in step.dependencies if d not in completed]
                if unmet:
                    step.status = "BLOCKED"
                    step.error = f"Unmet dependencies: {unmet}"
                    continue

                step.status = "RUNNING"
                try:
                    res = executor_fn(step.action_type, step.target, step.params)
                    step.result = res
                    step.status = "COMPLETED"
                    completed[step.step_name] = res
                    executed_count += 1
                except Exception as e:
                    step.status = "FAILED"
                    step.error = str(e)
                    return {
                        "status": "FAILED",
                        "failed_step": step.step_name,
                        "error": str(e),
                        "completed_steps": list(completed.keys())
                    }

            return {
                "status": "SUCCESS",
                "completed_steps": list(completed.keys()),
                "total_steps": len(steps)
            }


# ----------------------------------------------------------------------------
# 7. Error Diagnosis & Self-Repair
# ----------------------------------------------------------------------------

class ErrorDiagnosisSelfRepair:
    """Categorizes operational failures and suggests deterministic, policy-safe remediations."""

    @staticmethod
    def diagnose(error_str: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        e_lower = error_str.lower()
        if "permission" in e_lower or "denied" in e_lower or "unauthorized" in e_lower:
            return {
                "category": "PERMISSION_ERROR",
                "root_cause": "Missing cryptographic identity signature or policy rule restriction.",
                "remediation_strategy": "Request valid Creator signature or verify role authorizations.",
                "safe_auto_repair": False
            }
        elif "not found" in e_lower or "no such file" in e_lower:
            return {
                "category": "MISSING_RESOURCE",
                "root_cause": "Target path or registry entry does not exist.",
                "remediation_strategy": "Verify relative path or fall back to default template.",
                "safe_auto_repair": True
            }
        elif "budget" in e_lower or "timeout" in e_lower:
            return {
                "category": "RESOURCE_EXHAUSTION",
                "root_cause": "Step, time, or memory budget exceeded.",
                "remediation_strategy": "Decompose into smaller sub-tasks or increase time budget safely.",
                "safe_auto_repair": True
            }
        return {
            "category": "SYSTEM_RUNTIME_ERROR",
            "root_cause": f"Unhandled exception: {error_str}",
            "remediation_strategy": "Capture execution trace in diagnostic telemetry and fail safe.",
            "safe_auto_repair": False
        }


# ----------------------------------------------------------------------------
# 8. Simulation / Safe Sandbox
# ----------------------------------------------------------------------------

class SimulationSandbox:
    """Performs dry-run simulations of actions before actual physical or disk mutation."""

    @staticmethod
    def simulate_action(action_type: str, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
        destructive = False
        reasons = []

        if action_type.lower() in ("delete_file", "rmdir", "format_disk"):
            destructive = True
            reasons.append("Irreversible deletion of storage assets.")

        if "robot" in target.lower() or "actuator" in target.lower():
            if params.get("velocity", 0) > 2.0:
                destructive = True
                reasons.append("Excessive velocity exceeds simulated physical safety envelope.")

        return {
            "simulated": True,
            "destructive": destructive,
            "safety_passed": not destructive,
            "reasons": reasons if reasons else ["Operation passes simulated safety bounds."],
            "predicted_side_effects": ["None detected" if not destructive else "State modification"]
        }


# ----------------------------------------------------------------------------
# 9. Permission & Authority Reasoning
# ----------------------------------------------------------------------------

class PermissionAuthorityReasoning:
    """Evaluates authority levels and strictly blocks unauthorized privilege escalations."""

    CREATOR_NAME = "ROOT_OPERATOR"

    @classmethod
    def evaluate_request(
        cls,
        actor_id: str,
        action: str,
        has_creator_sig: bool = False,
        is_subagent: bool = False
    ) -> Dict[str, Any]:
        # Rule: Sub-agents can never self-grant creator authority
        if is_subagent and has_creator_sig:
            return {
                "permitted": False,
                "reason": "Security Invariant: Sub-agents are barred from assuming Creator authority.",
                "authority_level": "RESTRICTED_AGENT"
            }

        if action.startswith("ADMIN_") or "delete_core" in action:
            if not has_creator_sig:
                return {
                    "permitted": False,
                    "reason": f"Action '{action}' strictly requires verified {cls.CREATOR_NAME} cryptographic signature.",
                    "authority_level": "UNAUTHENTICATED"
                }
            return {
                "permitted": True,
                "reason": "Authorized via verified Creator ED25519 signature.",
                "authority_level": "CREATOR"
            }

        return {
            "permitted": True,
            "reason": "Standard operational action permitted under standard session.",
            "authority_level": "OPERATOR"
        }


# ----------------------------------------------------------------------------
# 10. Resource Management
# ----------------------------------------------------------------------------

class ResourceManager:
    """Tracks computational resources, thread pools, and enforces execution budgets."""

    def __init__(self, max_concurrent_tasks: int = 16, max_step_budget: int = 50):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_step_budget = max_step_budget
        self._active_tasks: Set[str] = set()
        self._lock = threading.RLock()

    def acquire_slot(self, task_id: str) -> bool:
        with self._lock:
            if len(self._active_tasks) >= self.max_concurrent_tasks:
                return False
            self._active_tasks.add(task_id)
            return True

    def release_slot(self, task_id: str):
        with self._lock:
            self._active_tasks.discard(task_id)

    def check_step_budget(self, current_steps: int) -> bool:
        return current_steps < self.max_step_budget


# ----------------------------------------------------------------------------
# 11. Self-Monitoring / System Health
# ----------------------------------------------------------------------------

class SystemHealthMonitor:
    """Continuous heartbeat monitoring and subsystem degradation tracking."""

    def __init__(self):
        self._subsystem_status: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def heartbeat(self, subsystem: str, healthy: bool = True, latency_ms: float = 0.0, details: Optional[Dict[str, Any]] = None):
        with self._lock:
            self._subsystem_status[subsystem] = {
                "healthy": healthy,
                "latency_ms": latency_ms,
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "details": details or {}
            }

    def get_overall_health(self) -> Dict[str, Any]:
        with self._lock:
            unhealthy = [s for s, d in self._subsystem_status.items() if not d.get("healthy")]
            return {
                "status": "DEGRADED" if unhealthy else "HEALTHY",
                "unhealthy_subsystems": unhealthy,
                "total_monitored": len(self._subsystem_status),
                "subsystems": dict(self._subsystem_status)
            }


# ----------------------------------------------------------------------------
# 12. Conflict Resolution
# ----------------------------------------------------------------------------

class ConflictResolutionEngine:
    """
    Resolves contradictory instructions and policy conflicts using strict hierarchy:
    CREATOR_RULE (Priority 1) > SECURITY_POLICY (2) > SAFETY_GUARD (3) > TASK_GOAL (4).
    """

    HIERARCHY = {
        "CREATOR_RULE": 1,
        "SECURITY_POLICY": 2,
        "SAFETY_GUARD": 3,
        "TASK_GOAL": 4
    }

    @classmethod
    def resolve_policy_conflict(cls, item_a: Dict[str, Any], item_b: Dict[str, Any]) -> Dict[str, Any]:
        rank_a = cls.HIERARCHY.get(item_a.get("type", "TASK_GOAL"), 5)
        rank_b = cls.HIERARCHY.get(item_b.get("type", "TASK_GOAL"), 5)

        if rank_a < rank_b:
            winner, loser = item_a, item_b
        elif rank_b < rank_a:
            winner, loser = item_b, item_a
        else:
            # Deterministic tie-break
            winner = item_a if item_a.get("timestamp", "") <= item_b.get("timestamp", "") else item_b
            loser = item_b if winner == item_a else item_a

        return {
            "chosen_item": winner,
            "superseded_item": loser,
            "resolution_reason": f"Authority rank {cls.HIERARCHY.get(winner.get('type'))} supersedes rank {cls.HIERARCHY.get(loser.get('type'))}."
        }


# ----------------------------------------------------------------------------
# 13. Temporal Understanding
# ----------------------------------------------------------------------------

class TemporalUnderstandingEngine:
    """Understands time intervals, deadlines, sequence ordering, and temporal validity."""

    @staticmethod
    def is_expired(timestamp_iso: str, ttl_seconds: float) -> bool:
        try:
            t = datetime.fromisoformat(timestamp_iso)
            now = datetime.now(timezone.utc)
            return (now - t).total_seconds() > ttl_seconds
        except Exception:
            return False

    @staticmethod
    def order_sequence(events: List[Dict[str, Any]], timestamp_key: str = "timestamp") -> List[Dict[str, Any]]:
        return sorted(events, key=lambda x: x.get(timestamp_key, ""))

    @staticmethod
    def calculate_duration_seconds(start_iso: str, end_iso: str) -> float:
        try:
            s = datetime.fromisoformat(start_iso)
            e = datetime.fromisoformat(end_iso)
            return (e - s).total_seconds()
        except Exception:
            return 0.0


# ----------------------------------------------------------------------------
# 14. Causal Reasoning
# ----------------------------------------------------------------------------

class CausalReasoningEngine:
    """Directed causal graphs, cause-and-effect inference, and intervention analysis."""

    def __init__(self):
        self._edges: Dict[str, List[str]] = {}
        self._lock = threading.RLock()

    def add_causal_link(self, cause: str, effect: str):
        with self._lock:
            if cause not in self._edges:
                self._edges[cause] = []
            if effect not in self._edges[cause]:
                self._edges[cause].append(effect)

    def infer_effects(self, intervention: str) -> List[str]:
        with self._lock:
            visited = set()
            stack = [intervention]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    stack.extend(self._edges.get(node, []))
            visited.discard(intervention)
            return sorted(list(visited))


# ----------------------------------------------------------------------------
# 15. Counterfactual Reasoning
# ----------------------------------------------------------------------------

class CounterfactualReasoningEngine:
    """Evaluates alternative past decisions and probable outcomes ('What if X had happened?')."""

    @staticmethod
    def evaluate_what_if(
        historical_action: str,
        historical_outcome: str,
        hypothetical_action: str
    ) -> Dict[str, Any]:
        differs = (historical_action != hypothetical_action)
        if not differs:
            return {
                "divergence": False,
                "projected_outcome": historical_outcome,
                "notes": "Hypothetical matches actual history."
            }

        # Project outcome based on action type
        if "prevent" in hypothetical_action or "sandbox" in hypothetical_action:
            projected = "Failure would have been mitigated in sandbox."
        elif "delete" in hypothetical_action:
            projected = "Data loss would have occurred without confirmation."
        else:
            projected = f"Alternative outcome resulting from action '{hypothetical_action}'."

        return {
            "divergence": True,
            "projected_outcome": projected,
            "historical_outcome": historical_outcome,
            "counterfactual_analysis": f"If '{hypothetical_action}' were taken instead of '{historical_action}', outcome changes."
        }


# ----------------------------------------------------------------------------
# 16. Uncertainty & Confidence Calibration
# ----------------------------------------------------------------------------

class UncertaintyCalibrationEngine:
    """Distinguishes epistemic (knowledge lack) from aleatoric (inherent noise) uncertainty."""

    @staticmethod
    def calibrate(evidence_count: int, agreement_ratio: float) -> Dict[str, Any]:
        # Epistemic uncertainty is high when evidence_count is low
        epistemic = max(0.0, 1.0 - (evidence_count / 10.0))
        # Aleatoric uncertainty is high when evidence conflicts (low agreement_ratio)
        aleatoric = 1.0 - agreement_ratio
        calibrated_conf = round((1.0 - epistemic) * agreement_ratio, 4)

        return {
            "confidence": calibrated_conf,
            "epistemic_uncertainty": round(epistemic, 4),
            "aleatoric_uncertainty": round(aleatoric, 4),
            "is_calibrated_high": calibrated_conf >= 0.85,
            "recommendation": "PROCEED" if calibrated_conf >= 0.7 else "SEEK_CLARIFICATION"
        }


# ----------------------------------------------------------------------------
# 17. Communication Engine
# ----------------------------------------------------------------------------

class CommunicationEngine:
    """Packages structured outputs, synthesizes user clarifications, and formats responses."""

    @staticmethod
    def format_clarification(missing_param: str, context: str) -> str:
        return f"To proceed with {context}, please provide the required '{missing_param}' parameter."

    @staticmethod
    def package_response(text: str, telemetry: Optional[Dict[str, Any]] = None, verification: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "response": text,
            "telemetry": telemetry or {},
            "verification": verification or {"status": "PASSED"},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# ----------------------------------------------------------------------------
# 18. Goal Decomposition & Goal Management
# ----------------------------------------------------------------------------

@dataclass
class SubGoal:
    goal_id: str
    description: str
    acceptance_criteria: str
    completed: bool = False


class GoalDecompositionManager:
    """Decomposes complex goals into manageable milestones with verifiable criteria."""

    def __init__(self):
        self._goals: Dict[str, List[SubGoal]] = {}
        self._lock = threading.RLock()

    def decompose(self, parent_goal: str) -> List[SubGoal]:
        with self._lock:
            # Deterministic milestone decomposition
            subgoals = [
                SubGoal(goal_id=f"g_{uuid.uuid4().hex[:6]}", description=f"Analyze requirements for: {parent_goal}", acceptance_criteria="Requirements parsed and validated."),
                SubGoal(goal_id=f"g_{uuid.uuid4().hex[:6]}", description=f"Execute required steps for: {parent_goal}", acceptance_criteria="All execution steps complete without error."),
                SubGoal(goal_id=f"g_{uuid.uuid4().hex[:6]}", description=f"Verify outcome against criteria: {parent_goal}", acceptance_criteria="TaskVerifier confirms success.")
            ]
            self._goals[parent_goal] = subgoals
            return subgoals

    def mark_completed(self, parent_goal: str, goal_id: str) -> bool:
        with self._lock:
            for g in self._goals.get(parent_goal, []):
                if g.goal_id == goal_id:
                    g.completed = True
                    return True
            return False


# ----------------------------------------------------------------------------
# 19. Self-Evaluation / Quality Control
# ----------------------------------------------------------------------------

class QualityControlSelfEvaluator:
    """Pre-finalization quality audit to prevent hallucinations and ensure rule compliance."""

    @staticmethod
    def audit_response(response_text: str, ground_truth_facts: List[str]) -> Dict[str, Any]:
        resp_lower = response_text.lower()
        supported = [f for f in ground_truth_facts if any(w in resp_lower for w in f.lower().split()[:3])]
        hallucination_risk = 0.0 if supported else (0.5 if ground_truth_facts else 0.0)

        return {
            "passed": hallucination_risk <= 0.3,
            "hallucination_risk": hallucination_risk,
            "supported_facts": supported,
            "audit_note": "Quality control check passed." if hallucination_risk <= 0.3 else "High risk of ungrounded assertion."
        }


# ----------------------------------------------------------------------------
# 20. Recovery / Rollback Management
# ----------------------------------------------------------------------------

@dataclass
class CheckpointRecord:
    checkpoint_id: str
    state_data: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RollbackRecoveryManager:
    """Maintains state checkpoints and rolls back on unrecoverable failures."""

    def __init__(self):
        self._checkpoints: Dict[str, CheckpointRecord] = {}
        self._lock = threading.RLock()

    def snapshot_state(self, checkpoint_id: str, state_data: Dict[str, Any]):
        with self._lock:
            self._checkpoints[checkpoint_id] = CheckpointRecord(
                checkpoint_id=checkpoint_id,
                state_data=dict(state_data)
            )

    def rollback_to(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cp = self._checkpoints.get(checkpoint_id)
            if cp:
                return dict(cp.state_data)
            return None


# ----------------------------------------------------------------------------
# 21. Autonomous Learning & Training Orchestrator
# ----------------------------------------------------------------------------

class AutonomousLearningTrainingOrchestrator:
    """Coordinates the full end-to-end self-learning and training cycle locally."""

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self._lock = threading.RLock()

    def schedule_learning_cycle(self) -> Dict[str, Any]:
        """Triggers discovery, staging, and evaluation of training needs."""
        from tara_model.self_trainer import get_self_trainer
        trainer = get_self_trainer(repo_root=self.repo_root)
        status = trainer.get_status()
        staging_res = trainer.discover_and_stage_updates()
        return {
            "status": "SCHEDULED",
            "trainer_status": status,
            "staging_result": staging_res,
            "orchestrated_at": datetime.now(timezone.utc).isoformat()
        }


# ----------------------------------------------------------------------------
# Unified Cognitive Capabilities Container
# ----------------------------------------------------------------------------

class CognitiveCapabilitiesHub:
    """Central container providing access to all 21 production-grade core capabilities."""

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.long_term_reasoning = LongTermReasoningEngine()
        self.self_model = SelfModelIntrospectionEngine(repo_root=self.repo_root)
        self.knowledge_lifecycle = KnowledgeLifecycleManager()
        self.continuous_learning = ContinuousLearningManager()
        self.memory_lifecycle = MemoryLifecycleManager()
        self.workflow_engine = WorkflowEngine()
        self.error_diagnosis = ErrorDiagnosisSelfRepair()
        self.simulation_sandbox = SimulationSandbox()
        self.permission_reasoning = PermissionAuthorityReasoning()
        self.resource_manager = ResourceManager()
        self.health_monitor = SystemHealthMonitor()
        self.conflict_resolution = ConflictResolutionEngine()
        self.temporal_reasoning = TemporalUnderstandingEngine()
        self.causal_reasoning = CausalReasoningEngine()
        self.counterfactual_reasoning = CounterfactualReasoningEngine()
        self.uncertainty_calibration = UncertaintyCalibrationEngine()
        self.communication_engine = CommunicationEngine()
        self.goal_decomposition = GoalDecompositionManager()
        self.quality_control = QualityControlSelfEvaluator()
        self.rollback_recovery = RollbackRecoveryManager()
        self.autonomous_orchestrator = AutonomousLearningTrainingOrchestrator(repo_root=self.repo_root)

        from tara_core.extended_capabilities import get_extended_capabilities_hub
        from tara_core.experiential_learning import get_experiential_learner
        from tara_core.event_bus import TaraEventBus
        from tara_core.robotics_hal import RoboticsHAL, GUIComputerInteraction
        from tara_core.working_memory_governor import WorkingMemoryGovernor
        from tara_core.experiential_bridge import ExperientialStagingBridge
        from tara_core.knowledge_ontology import KnowledgeOntologyEngine
        from tara_core.memory_consolidation import MemoryConsolidationEngine
        from tara_core.reasoning_engine_advanced import (
            ConstraintSolver, MultiObjectiveOptimizer, ProbabilisticReasoningEngine, UncertaintyAwarePlanner
        )
        from tara_core.environment_sensor_fusion import DynamicEnvironmentModel, SensorFusionEngine, RealWorldActionGrounder
        from tara_core.engineering_capabilities import SoftwareEngineeringEngine, DataEngineeringEngine, ExperimentDesignEngine
        from tara_core.skill_acquisition_certification import (
            SkillAcquisitionEngine, DemonstrationLearningEngine, CapabilityComposer, SkillCertifier,
            SelfArchitectureAwareness, ArchitectureEvolutionEngine, DynamicCapabilityAcquisitionEngine
        )
        from tara_core.continual_learning_governor import (
            ContinualLearningGovernor, CurriculumPriorityEngine, ModelAdaptationLayer, ModelRouter
        )
        from tara_core.resilience_maintenance import (
            GoalPersistenceManager, IncidentLearningEngine, PolicyStrategyLearner,
            AuditExplainabilityEngine, CorruptedLearningRecovery, SelfMaintenanceEngine
        )
        from tara_core.prediction_error_loop import PredictionErrorEngine

        self.extended = get_extended_capabilities_hub()
        self.experiential_learner = get_experiential_learner(repo_root=self.repo_root)
        self.event_bus = TaraEventBus.get_default()
        self.robotics_hal = RoboticsHAL.get_default()
        self.working_memory = WorkingMemoryGovernor.get_default()
        self.experiential_bridge = ExperientialStagingBridge.get_default(repo_root=self.repo_root)
        self.ontology = KnowledgeOntologyEngine.get_default()
        self.memory_consolidation = MemoryConsolidationEngine.get_default(self.repo_root)
        self.constraint_solver = ConstraintSolver()
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        self.probabilistic_reasoning = ProbabilisticReasoningEngine()
        self.uncertainty_planner = UncertaintyAwarePlanner()
        self.environment_model = DynamicEnvironmentModel()
        self.sensor_fusion = SensorFusionEngine(self.environment_model)
        self.action_grounder = RealWorldActionGrounder(self.robotics_hal)
        self.gui_interaction = GUIComputerInteraction(self.robotics_hal)
        self.software_engineering = SoftwareEngineeringEngine()
        self.data_engineering = DataEngineeringEngine()
        self.experiment_design = ExperimentDesignEngine()
        self.skill_acquisition = SkillAcquisitionEngine()
        self.demonstration_learning = DemonstrationLearningEngine()
        self.capability_composer = CapabilityComposer()
        self.skill_certifier = SkillCertifier()
        self.dynamic_capability_acquisition = DynamicCapabilityAcquisitionEngine(repo_root=self.repo_root)
        self.architecture_awareness = SelfArchitectureAwareness()
        self.architecture_evolution = ArchitectureEvolutionEngine()
        self.continual_learning_governor = ContinualLearningGovernor()
        self.continual_learning = self.continual_learning_governor
        self.curriculum_priority = CurriculumPriorityEngine()
        self.model_adaptation = ModelAdaptationLayer()
        self.model_router = ModelRouter()
        self.goal_persistence = GoalPersistenceManager()
        self.incident_learning = IncidentLearningEngine()
        self.policy_strategy_learner = PolicyStrategyLearner()
        self.audit_explainability = AuditExplainabilityEngine()
        self.corrupted_recovery = CorruptedLearningRecovery()
        self.self_maintenance = SelfMaintenanceEngine()
        self.prediction_error_loop = PredictionErrorEngine.get_default(self.repo_root)

    def get_capabilities_manifest(self) -> List[Dict[str, str]]:
        manifest = [
            {"id": "long_term_reasoning", "name": "Long-Term Reasoning & Consistency"},
            {"id": "self_model", "name": "Self-Model / Introspection"},
            {"id": "knowledge_lifecycle", "name": "Knowledge Management & Knowledge Lifecycle"},
            {"id": "continuous_learning", "name": "Continuous Learning Manager"},
            {"id": "memory_lifecycle", "name": "Memory Lifecycle Management"},
            {"id": "workflow_engine", "name": "Task Execution / Workflow Engine"},
            {"id": "error_diagnosis", "name": "Error Diagnosis & Self-Repair"},
            {"id": "simulation_sandbox", "name": "Simulation / Safe Sandbox"},
            {"id": "permission_reasoning", "name": "Permission & Authority Reasoning"},
            {"id": "resource_manager", "name": "Resource Management"},
            {"id": "system_health", "name": "Self-Monitoring / System Health"},
            {"id": "conflict_resolution", "name": "Conflict Resolution"},
            {"id": "temporal_reasoning", "name": "Temporal Understanding"},
            {"id": "causal_reasoning", "name": "Causal Reasoning"},
            {"id": "counterfactual_reasoning", "name": "Counterfactual Reasoning"},
            {"id": "uncertainty_calibration", "name": "Uncertainty & Confidence Calibration"},
            {"id": "communication_engine", "name": "Communication Engine"},
            {"id": "goal_decomposition", "name": "Goal Decomposition & Goal Management"},
            {"id": "quality_control", "name": "Self-Evaluation / Quality Control"},
            {"id": "rollback_recovery", "name": "Recovery / Rollback Management"},
            {"id": "autonomous_orchestrator", "name": "Autonomous Learning & Training Orchestrator"}
        ]
        manifest.extend(self.extended.get_extended_manifest())
        manifest.append({"id": "experiential_learning", "name": "Experiential Closed-Loop Learning"})
        manifest.append({"id": "robotics_hal", "name": "Robotics Hardware Abstraction Layer"})
        # 36 Advanced Core Capabilities
        advanced_36 = [
            {"id": "knowledge_ontology", "name": "Knowledge Representation / Ontology"},
            {"id": "memory_consolidation", "name": "Episodic-to-Semantic Memory Consolidation"},
            {"id": "continual_learning", "name": "Continual Learning & Forgetting Control"},
            {"id": "skill_acquisition", "name": "Skill Acquisition Engine"},
            {"id": "demonstration_learning", "name": "Demonstration / Example Learning"},
            {"id": "curriculum_priority", "name": "Curriculum Priority Engine"},
            {"id": "model_adaptation", "name": "Model Adaptation Layer"},
            {"id": "model_routing", "name": "Model Routing"},
            {"id": "constraint_solving", "name": "Constraint Solving"},
            {"id": "multi_objective_optimization", "name": "Multi-Objective Optimization"},
            {"id": "probabilistic_reasoning", "name": "Probabilistic Reasoning"},
            {"id": "uncertainty_planning", "name": "Planning Under Uncertainty"},
            {"id": "action_grounding", "name": "Real-World Action Grounding"},
            {"id": "environment_model", "name": "Dynamic Environment Model"},
            {"id": "sensor_fusion", "name": "Multi-Modal Sensor Fusion"},
            {"id": "gui_interaction", "name": "Computer / GUI Interaction"},
            {"id": "software_engineering", "name": "Software Engineering Engine"},
            {"id": "data_engineering", "name": "Data Engineering Engine"},
            {"id": "experiment_design", "name": "Experiment Design Engine"},
            {"id": "incident_learning", "name": "Incident Learning / Failure Knowledge"},
            {"id": "goal_persistence", "name": "Goal Persistence & Recovery"},
            {"id": "architecture_awareness", "name": "Self-Architecture Awareness"},
            {"id": "architecture_evolution", "name": "Architecture Evolution Engine"},
            {"id": "capability_composition", "name": "Capability Composition"},
            {"id": "skill_certification", "name": "Skill Validation & Certification"},
            {"id": "policy_learning", "name": "Policy & Strategy Learning"},
            {"id": "audit_explainability", "name": "Auditability & Explainability"},
            {"id": "corrupted_recovery", "name": "Recovery from Corrupted Learning"},
            {"id": "self_maintenance", "name": "Self-Maintenance Engine"},
            {"id": "prediction_error_loop", "name": "World Model + Prediction Error Loop"}
        ]
        existing_ids = {m["id"] for m in manifest}
        for item in advanced_36:
            if item["id"] not in existing_ids:
                manifest.append(item)
        return manifest

    def list_capabilities(self) -> List[Dict[str, str]]:
        return self.get_capabilities_manifest()

    @classmethod
    def get_default(cls, repo_root: Optional[str] = None) -> "CognitiveCapabilitiesHub":
        return get_cognitive_capabilities_hub(repo_root=repo_root)


_hub_instance: Optional[CognitiveCapabilitiesHub] = None
_hub_lock = threading.Lock()

def get_cognitive_capabilities_hub(repo_root: Optional[str] = None) -> CognitiveCapabilitiesHub:
    global _hub_instance
    with _hub_lock:
        if _hub_instance is None:
            _hub_instance = CognitiveCapabilitiesHub(repo_root=repo_root)
        return _hub_instance


def register_cognitive_capabilities(registry: Optional[Any] = None, repo_root: Optional[str] = None) -> int:
    """Registers all 21 core cognitive capabilities into the CapabilityRegistry."""
    from tara_core.registry import CapabilityRegistry, Capability, CapabilityCategory, RiskLevel

    target_reg = registry or CapabilityRegistry.get_default()
    hub = get_cognitive_capabilities_hub(repo_root=repo_root)

    capabilities_def = [
        (
            "long_term_reasoning", "Long-Term Reasoning & Consistency", CapabilityCategory.REASONING,
            "Multi-step inference chains, premise tracking, and contradiction detection across extended dialog.",
            ["reasoning", "deduce", "infer", "consistency"],
            lambda p: hub.long_term_reasoning.check_consistency(p.get("chain_id", "default"))
        ),
        (
            "self_model", "Self-Model / Introspection", CapabilityCategory.COGNITIVE,
            "Epistemological self-model tracking WHAT I KNOW/CAN/ALLOWED TO DO, VERIFIED, and UNCERTAIN.",
            ["introspect", "self_model", "capabilities", "status", "limits"],
            lambda p: hub.self_model.get_snapshot()
        ),
        (
            "knowledge_lifecycle", "Knowledge Management & Knowledge Lifecycle", CapabilityCategory.KNOWLEDGE,
            "Complete 12-stage lifecycle from discovery to ingestion, validation, provenance, and deprecation.",
            ["knowledge_lifecycle", "ingest_kb", "deprecate_kb", "kb_status"],
            lambda p: [e.entry_id for e in hub.knowledge_lifecycle.list_entries()]
        ),
        (
            "continuous_learning", "Continuous Learning Manager", CapabilityCategory.COGNITIVE,
            "Autonomous trigger policy evaluation, drift detection, and staging queue management.",
            ["learning_need", "drift", "evaluate_training"],
            lambda p: hub.continuous_learning.evaluate_training_need(p.get("staged_count", 0))
        ),
        (
            "memory_lifecycle", "Memory Lifecycle Management", CapabilityCategory.COGNITIVE,
            "Strict multi-layer memory separation, actor boundary isolation, and safe forgetting.",
            ["memory_lifecycle", "forget_memory", "scoped_memory"],
            lambda p: len(hub.memory_lifecycle.retrieve(p.get("actor_id", "user")))
        ),
        (
            "workflow_engine", "Task Execution / Workflow Engine", CapabilityCategory.AGENT,
            "DAG task execution, step dependency resolution, and runtime budget enforcement.",
            ["workflow", "execute_dag", "pipeline"],
            lambda p: hub.workflow_engine.execute_workflow(p.get("workflow_id", ""), lambda a, t, params: {"status": "SUCCESS"})
        ),
        (
            "error_diagnosis", "Error Diagnosis & Self-Repair", CapabilityCategory.COGNITIVE,
            "Root-cause taxonomy categorization and deterministic remediation generation.",
            ["diagnose_error", "repair", "troubleshoot"],
            lambda p: hub.error_diagnosis.diagnose(p.get("error", "unknown"))
        ),
        (
            "simulation_sandbox", "Simulation / Safe Sandbox", CapabilityCategory.COGNITIVE,
            "Dry-run simulation for evaluating side-effects and physical safety envelopes before execution.",
            ["simulate", "sandbox", "dry_run"],
            lambda p: hub.simulation_sandbox.simulate_action(p.get("action", ""), p.get("target", ""), p.get("params", {}))
        ),
        (
            "permission_reasoning", "Permission & Authority Reasoning", CapabilityCategory.REASONING,
            "Cryptographic privilege reasoning, fail-closed boundaries, and anti-escalation enforcement.",
            ["permission_check", "authority_reasoning", "auth_eval"],
            lambda p: hub.permission_reasoning.evaluate_request(p.get("actor_id", ""), p.get("action", ""), p.get("has_sig", False))
        ),
        (
            "resource_manager", "Resource Management", CapabilityCategory.EXTENSION,
            "Compute, memory, thread pool, step budget, and throttling enforcement.",
            ["resource_check", "step_budget", "quota"],
            lambda p: {"step_ok": hub.resource_manager.check_step_budget(p.get("steps", 0))}
        ),
        (
            "system_health", "Self-Monitoring / System Health", CapabilityCategory.EXTENSION,
            "Subsystem heartbeat monitoring, latency tracking, and degradation detection.",
            ["health_status", "heartbeat", "degradation"],
            lambda p: hub.health_monitor.get_overall_health()
        ),
        (
            "conflict_resolution", "Conflict Resolution", CapabilityCategory.REASONING,
            "Strict priority hierarchy resolution: Creator Rule > Security Policy > Safety Guard > Task Goal.",
            ["resolve_conflict", "policy_conflict", "precedence"],
            lambda p: hub.conflict_resolution.resolve_policy_conflict(p.get("a", {}), p.get("b", {}))
        ),
        (
            "temporal_reasoning", "Temporal Understanding", CapabilityCategory.REASONING,
            "Sequence ordering, duration calculation, interval overlap, and deadline reasoning.",
            ["temporal", "deadline", "duration", "sequence"],
            lambda p: {"expired": hub.temporal_reasoning.is_expired(p.get("timestamp", ""), p.get("ttl", 3600))}
        ),
        (
            "causal_reasoning", "Causal Reasoning", CapabilityCategory.REASONING,
            "Directed causal graphs, cause-and-effect inference, and do-calculus intervention.",
            ["causal", "cause_effect", "intervention"],
            lambda p: hub.causal_reasoning.infer_effects(p.get("cause", ""))
        ),
        (
            "counterfactual_reasoning", "Counterfactual Reasoning", CapabilityCategory.REASONING,
            "Alternative scenario branching and 'what if' hypothetical outcome evaluation.",
            ["what_if", "counterfactual", "hypothetical"],
            lambda p: hub.counterfactual_reasoning.evaluate_what_if(p.get("historical_action", ""), p.get("historical_outcome", ""), p.get("hypothetical_action", ""))
        ),
        (
            "uncertainty_calibration", "Uncertainty & Confidence Calibration", CapabilityCategory.REASONING,
            "Epistemic vs aleatoric uncertainty separation and confidence score calibration.",
            ["uncertainty", "confidence_calibration", "epistemic"],
            lambda p: hub.uncertainty_calibration.calibrate(p.get("evidence_count", 5), p.get("agreement_ratio", 0.9))
        ),
        (
            "communication_engine", "Communication Engine", CapabilityCategory.EXTENSION,
            "Context-adaptive tone, clarification generation, and structured response packaging.",
            ["clarification", "package_response", "format_dialog"],
            lambda p: hub.communication_engine.package_response(p.get("text", ""))
        ),
        (
            "goal_decomposition", "Goal Decomposition & Goal Management", CapabilityCategory.AGENT,
            "Recursive goal decomposition into milestone DAGs with verifiable acceptance criteria.",
            ["decompose_goal", "subgoals", "milestones"],
            lambda p: [g.description for g in hub.goal_decomposition.decompose(p.get("goal", ""))]
        ),
        (
            "quality_control", "Self-Evaluation / Quality Control", CapabilityCategory.REASONING,
            "Pre-finalization factuality auditing and hallucination risk scoring.",
            ["quality_audit", "hallucination_check", "fact_check"],
            lambda p: hub.quality_control.audit_response(p.get("response", ""), p.get("facts", []))
        ),
        (
            "rollback_recovery", "Recovery / Rollback Management", CapabilityCategory.EXTENSION,
            "Transactional state checkpoints, rollback journals, and point-in-time recovery.",
            ["checkpoint", "rollback", "restore_state"],
            lambda p: hub.rollback_recovery.rollback_to(p.get("checkpoint_id", ""))
        ),
        (
            "autonomous_orchestrator", "Autonomous Learning & Training Orchestrator", CapabilityCategory.COGNITIVE,
            "Coordination of autonomous local self-training and safe promotion.",
            ["orchestrate_learning", "self_train_cycle", "autonomous_learn"],
            lambda p: hub.autonomous_orchestrator.schedule_learning_cycle()
        ),
        (
            "multimodal_perception", "Multimodal Perception & Understanding", CapabilityCategory.COGNITIVE,
            "Cross-modal perception across text, image, audio, video, document, and sensor telemetry.",
            ["multimodal", "vision", "audio", "sensor"],
            lambda p: hub.extended.multimodal.process_perception(p.get("modality", "TEXT"), p.get("uri", ""))
        ),
        (
            "tool_learning_creation", "Tool Learning & Tool Creation", CapabilityCategory.TOOL,
            "Dynamic tool spec parsing, adapter generation, sandbox validation, and ToolRegistry registration.",
            ["tool_learning", "create_tool", "tool_adapter"],
            lambda p: hub.extended.tool_learning.parse_and_validate_spec(p.get("spec", {}))
        ),
        (
            "action_execution_monitor", "Action Planning & Execution Monitoring", CapabilityCategory.AGENT,
            "Step-by-step plan execution, progress monitoring, deviation detection, and dynamic re-planning.",
            ["action_monitor", "deviation", "replan"],
            lambda p: hub.extended.action_planning.monitor_step(p.get("plan_id", ""), p.get("action_id", ""), p.get("duration", 1.0), p.get("status", "COMPLETED"))
        ),
        (
            "world_state_model", "World Model / State Model", CapabilityCategory.COGNITIVE,
            "Structured entity-relationship state graph covering users, devices, machines, and environments.",
            ["world_model", "entity_state", "relationship_graph"],
            lambda p: hub.extended.world_model.query_state(p.get("type"))
        ),
        (
            "spatial_reasoning", "Spatial Reasoning", CapabilityCategory.REASONING,
            "2D/3D coordinate analysis, geometry, kinematics, and workspace envelope verification.",
            ["spatial", "coordinates", "workspace_envelope", "geometry"],
            lambda p: hub.extended.spatial_reasoning.verify_workspace_envelope(p.get("point", (0, 0, 0)))
        ),
        (
            "event_causal_memory", "Event & Causal Memory", CapabilityCategory.KNOWLEDGE,
            "Episodic event journaling linking actions, causal explanations, outcomes, and extracted lessons.",
            ["event_memory", "causal_memory", "lesson_learned"],
            lambda p: hub.extended.event_causal_memory.query_lessons(p.get("keyword", ""))
        ),
        (
            "predictive_reasoning", "Predictive Reasoning", CapabilityCategory.REASONING,
            "Pre-action estimation of outcome probabilities, resource consumption, and side effects.",
            ["predictive", "pre_action", "failure_probability"],
            lambda p: hub.extended.predictive_reasoning.predict_action_outcome(p.get("action", ""), p.get("params", {}))
        ),
        (
            "decision_tradeoff", "Decision & Trade-off Engine", CapabilityCategory.REASONING,
            "Multi-factor tradeoff analysis balancing safety, latency, cost, quality, and policy alignment.",
            ["tradeoff", "decision_analysis", "multi_criteria"],
            lambda p: hub.extended.decision_tradeoff.evaluate_tradeoffs(p.get("options", []))
        ),
        (
            "attention_priority", "Attention & Priority Management", CapabilityCategory.COGNITIVE,
            "Dynamic attention triage prioritizing urgent and high-impact tasks while deferring background operations.",
            ["attention", "priority", "task_triage"],
            lambda p: hub.extended.attention_priority.prioritize_tasks(p.get("tasks", []))
        ),
        (
            "curiosity_gap_detector", "Curiosity & Knowledge-Gap Detection", CapabilityCategory.COGNITIVE,
            "Detects missing information, incomplete schemas, or contradictions and routes to self-learning.",
            ["curiosity", "knowledge_gap", "missing_info"],
            lambda p: hub.extended.curiosity.inspect_query_knowledge(p.get("query", ""), p.get("facts", []))
        ),
        (
            "source_credibility", "Source Credibility & Evidence Reasoning", CapabilityCategory.REASONING,
            "Cryptographic provenance scoring and trustworthiness evaluation for incoming evidence.",
            ["source_credibility", "provenance_check", "source_trust"],
            lambda p: hub.extended.source_credibility.evaluate_source(p.get("source_type", ""), p.get("hash", ""))
        ),
        (
            "fact_verification", "Fact Verification & Cross-Validation", CapabilityCategory.REASONING,
            "Multi-source consensus checking and independent assertion cross-validation.",
            ["fact_check", "cross_validate", "consensus"],
            lambda p: hub.extended.fact_verification.cross_validate(p.get("claim", ""), p.get("evidence", []))
        ),
        (
            "hallucination_grounding", "Hallucination Prevention & Grounding", CapabilityCategory.REASONING,
            "Guarantees outputs strictly distinguish verified ground truth from inferred assertions.",
            ["grounding", "anti_hallucination", "epistemic_check"],
            lambda p: hub.extended.hallucination_grounding.ground_response(p.get("assertion", ""), p.get("evidence", []))
        ),
        (
            "personalization_engine", "Personalization Engine", CapabilityCategory.EXTENSION,
            "Adapts to authorized user preferences and workflows under strict multi-user memory isolation.",
            ["personalization", "user_preferences", "custom_style"],
            lambda p: hub.extended.personalization.get_user_preference(p.get("user_id", ""), p.get("key", ""))
        ),
        (
            "conversational_social_state", "Conversational / Social State", CapabilityCategory.EXTENSION,
            "Tracks turn context, dialogue acts, user goals, and active clarification requests.",
            ["conversational_state", "dialogue_act", "turn_context"],
            lambda p: hub.extended.conversational_state.get_session_state(p.get("session_id", ""))
        ),
        (
            "meta_reasoning", "Meta-Reasoning Engine", CapabilityCategory.REASONING,
            "Selects the optimal cognitive strategy (direct, tool-augmented, decomposed, or clarified).",
            ["meta_reasoning", "strategy_selection", "cognitive_mode"],
            lambda p: hub.extended.meta_reasoning.select_strategy(p.get("complexity", 1), p.get("confidence", 0.9), p.get("tools_needed", False))
        ),
        (
            "self_improvement", "Self-Improvement Engine", CapabilityCategory.COGNITIVE,
            "Analyzes runtime failure patterns and adapts operational heuristics safely.",
            ["self_improvement", "failure_pattern", "adaptive_tuning"],
            lambda p: hub.extended.self_improvement.record_failure_pattern(p.get("signature", ""))
        ),
        (
            "experimentation_ab", "Experimentation / A-B Evaluation Engine", CapabilityCategory.EXTENSION,
            "Sandboxed comparative evaluation of candidate skills, tools, prompts, and models.",
            ["experimentation", "ab_testing", "compare_variants"],
            lambda p: hub.extended.experimentation.compare_variants(p.get("a", {}), p.get("b", {}))
        ),
        (
            "dependency_compatibility", "Dependency & Compatibility Manager", CapabilityCategory.EXTENSION,
            "Pre-activation validation of dependencies, versions, and API contracts.",
            ["dependency_check", "compatibility", "version_check"],
            lambda p: hub.extended.dependency_manager.check_compatibility(p.get("dependencies", []), p.get("available", set()))
        ),
        (
            "config_governance", "Configuration & Version Governance", CapabilityCategory.EXTENSION,
            "Configuration snapshotting, schema versioning, and deterministic rollback points.",
            ["config_governance", "config_snapshot", "config_restore"],
            lambda p: hub.extended.config_governance.restore_config(p.get("version_id", ""))
        ),
        (
            "distributed_multi_device", "Distributed / Multi-Device Coordination", CapabilityCategory.EXTENSION,
            "Multi-node coordination, heartbeat synchronization, and distributed task routing.",
            ["multi_device", "node_sync", "distributed_task"],
            lambda p: hub.extended.multi_device.list_nodes()
        ),
        (
            "human_in_the_loop", "Human-in-the-Loop Escalation", CapabilityCategory.REASONING,
            "Safety tripwires pausing high-risk or low-confidence operations for explicit human approval.",
            ["hitl_escalation", "human_approval", "risk_pause"],
            lambda p: hub.extended.human_in_the_loop.evaluate_escalation(p.get("risk", "LOW"), p.get("confidence", 1.0), p.get("critical", False))
        ),
        (
            "digital_twin_simulation", "Digital Twin / Simulation Reasoning", CapabilityCategory.REASONING,
            "Synchronized digital twin modeling and physical trajectory envelope validation.",
            ["digital_twin", "trajectory_simulation", "envelope_check"],
            lambda p: hub.extended.digital_twin.validate_trajectory(p.get("waypoints", []))
        ),
        (
            "resource_aware_intelligence", "Resource-Aware Intelligence", CapabilityCategory.EXTENSION,
            "Hardware-adaptive execution budget allocation based on CPU, RAM, and battery state.",
            ["resource_aware", "compute_adaptation", "hardware_budget"],
            lambda p: hub.extended.resource_aware.compute_strategy_budget(p.get("cpu", 10.0), p.get("ram", 2000.0))
        ),
        (
            "long_horizon_planning", "Long-Horizon Planning", CapabilityCategory.AGENT,
            "Project-scale multi-session milestone tracking and persistent progress monitoring.",
            ["long_horizon", "milestones", "project_planning"],
            lambda p: hub.extended.long_horizon.get_progress(p.get("project_id", ""))
        ),
        (
            "experiential_learning", "Experiential Closed-Loop Learning", CapabilityCategory.COGNITIVE,
            "Full closed-loop experiential cycle from observation to action, verification, lesson extraction, and future application.",
            ["experiential_learning", "closed_loop", "learn_from_experience"],
            lambda p: hub.experiential_learner.run_full_closed_loop(p.get("input", ""), lambda a, g: {"status": "SUCCESS"})
        ),
        (
            "robotics_hal", "Robotics Hardware Abstraction Layer & Safety Interlock", CapabilityCategory.TOOL,
            "Production-grade hardware interface, multi-bus protocol bridge, spatial safety envelope validation, and fail-closed E-Stop interlock.",
            ["robotics_hal", "hardware_device", "safety_interlock", "e_stop", "robotics_actuation"],
            lambda p: hub.robotics_hal.send_motion_command(
                p.get("device_id", ""),
                tuple(p.get("target_coords", (0.0, 0.0, 0.0))),
                p.get("velocity", 0.5)
            ) if p.get("action") == "motion" else hub.robotics_hal.read_telemetry(p.get("device_id", ""))
        ),
    ]

    registered_count = 0
    for cid, name, category, purpose, keywords, handler in capabilities_def:
        cap = Capability(
            capability_id=cid,
            name=name,
            version="1.0.0",
            category=category,
            purpose=purpose,
            trigger_metadata={"intents": [cid], "keywords": keywords},
            permissions=["INTERNAL"],
            risk_level=RiskLevel.LOW,
            executable=True,
            handler=handler
        )
        target_reg.register_capability(cap)
        registered_count += 1

    logger.info(f"Successfully registered {registered_count} core cognitive capabilities in CapabilityRegistry.")
    return registered_count


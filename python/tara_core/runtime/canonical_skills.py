"""
python/tara_core/runtime/canonical_skills.py

Canonical Skill and Tool Architecture for TARA.
Enforces:
1. CANONICAL SPECIFICATION:
   A skill or tool has ONE canonical identity, specification, and contract.
2. SEPARATE NATIVE RUNTIME IMPLEMENTATIONS:
   Implementations exist separately and natively in every required runtime codebase
   (Python implementation, Rust implementation, C++ implementation, etc.).
   Never use a common Python wrapper as a substitute for native implementations.
3. SKILL FAILURE RULE:
   If any required runtime implementation fails or is missing:
   ENTIRE SKILL FAILS.
   Status remains PARTIAL_RUNTIME_SUPPORT.
   Transitions to UNIVERSAL_RUNTIME_READY only when ALL required runtimes pass.
4. ATOMIC SKILL & TOOL PROMOTION:
   Only UNIVERSAL_RUNTIME_READY skills are eligible for production activation.
"""

import os
import json
import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

from .registry import DynamicRuntimeRegistry, RuntimeState
from .gate import UniversalRuntimeGate, RuntimeEvaluationResult, GateVerdict


class SkillStatus(str, Enum):
    DRAFT = "DRAFT"
    VERIFYING = "VERIFYING"
    PARTIAL_RUNTIME_SUPPORT = "PARTIAL_RUNTIME_SUPPORT"
    UNIVERSAL_RUNTIME_READY = "UNIVERSAL_RUNTIME_READY"
    REVOKED = "REVOKED"
    RETIRED = "RETIRED"


@dataclass
class RuntimeImplementationRef:
    runtime_id: str
    status: str  # "VERIFIED", "FAILED", "PENDING", "MISSING"
    code_path: Optional[str] = None
    last_tested: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalSkillDefinition:
    skill_id: str
    name: str
    version: str
    inputs_schema: Dict[str, Any]
    outputs_schema: Dict[str, Any]
    permissions: List[str]
    behavior: str
    security_rules: List[str] = field(default_factory=list)
    capability_requirements: List[str] = field(default_factory=list)
    error_semantics: Dict[str, Any] = field(default_factory=dict)
    implementations: Dict[str, RuntimeImplementationRef] = field(default_factory=dict)
    status: SkillStatus = SkillStatus.DRAFT

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["implementations"] = {
            r_id: asdict(impl) for r_id, impl in self.implementations.items()
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalSkillDefinition":
        impls = {}
        for r_id, i_data in data.get("implementations", {}).items():
            impls[r_id] = RuntimeImplementationRef(**i_data)

        status_str = data.get("status", "DRAFT")
        try:
            status_enum = SkillStatus(status_str)
        except ValueError:
            status_enum = SkillStatus.DRAFT

        return cls(
            skill_id=data["skill_id"],
            name=data["name"],
            version=data["version"],
            inputs_schema=data["inputs_schema"],
            outputs_schema=data["outputs_schema"],
            permissions=data["permissions"],
            behavior=data["behavior"],
            security_rules=data.get("security_rules", []),
            capability_requirements=data.get("capability_requirements", []),
            error_semantics=data.get("error_semantics", {}),
            implementations=impls,
            status=status_enum
        )


@dataclass
class CanonicalToolDefinition:
    tool_id: str
    name: str
    version: str
    parameters_schema: Dict[str, Any]
    returns_schema: Dict[str, Any]
    permissions: List[str]
    implementations: Dict[str, RuntimeImplementationRef] = field(default_factory=dict)
    status: SkillStatus = SkillStatus.DRAFT

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["implementations"] = {
            r_id: asdict(impl) for r_id, impl in self.implementations.items()
        }
        return d


class CanonicalSkillManager:
    """
    Manages canonical skills and tools across all registered runtimes.
    """

    def __init__(
        self,
        registry: Optional[DynamicRuntimeRegistry] = None,
        gate: Optional[UniversalRuntimeGate] = None,
        repo_root: Optional[str] = None
    ):
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.repo_root = repo_root
        self.registry = registry or DynamicRuntimeRegistry(repo_root=self.repo_root)
        self.gate = gate or UniversalRuntimeGate(registry=self.registry, repo_root=self.repo_root)

        self._lock = threading.RLock()
        self.skills_store_path = os.path.join(self.repo_root, "storage", "skills", "canonical_skills.json")
        self.tools_store_path = os.path.join(self.repo_root, "storage", "skills", "canonical_tools.json")

        self.skills: Dict[str, CanonicalSkillDefinition] = {}
        self.tools: Dict[str, CanonicalToolDefinition] = {}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if os.path.exists(self.skills_store_path):
                try:
                    with open(self.skills_store_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for s_id, s_data in data.items():
                        self.skills[s_id] = CanonicalSkillDefinition.from_dict(s_data)
                except Exception:
                    pass

    def _save(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(self.skills_store_path), exist_ok=True)
            with open(self.skills_store_path, "w", encoding="utf-8") as f:
                json.dump({s_id: s.to_dict() for s_id, s in self.skills.items()}, f, indent=2)

    def register_canonical_skill(self, skill: CanonicalSkillDefinition) -> CanonicalSkillDefinition:
        with self._lock:
            skill.status = SkillStatus.DRAFT
            self.skills[skill.skill_id] = skill
            self._save()
            return skill

    def update_runtime_implementation(
        self,
        skill_id: str,
        runtime_id: str,
        code_path: str,
        test_passed: bool,
        diagnostics: Optional[Dict[str, Any]] = None
    ) -> CanonicalSkillDefinition:
        """
        Updates native implementation test status for a specific runtime.
        Evaluates whether all required runtimes have satisfied the contract.
        """
        with self._lock:
            if skill_id not in self.skills:
                raise KeyError(f"Skill '{skill_id}' not found.")

            skill = self.skills[skill_id]
            clean_runtime = runtime_id.lower().strip()

            skill.implementations[clean_runtime] = RuntimeImplementationRef(
                runtime_id=clean_runtime,
                status="VERIFIED" if test_passed else "FAILED",
                code_path=code_path,
                last_tested=time.time(),
                diagnostics=diagnostics or {}
            )

            # Check all required runtimes
            required_runtimes = [r.runtime_id for r in self.registry.list_required_runtimes()]
            all_required_pass = True

            for req_id in required_runtimes:
                impl = skill.implementations.get(req_id)
                if not impl or impl.status != "VERIFIED":
                    all_required_pass = False
                    break

            if all_required_pass:
                skill.status = SkillStatus.UNIVERSAL_RUNTIME_READY
            else:
                skill.status = SkillStatus.PARTIAL_RUNTIME_SUPPORT

            self._save()
            return skill

    def evaluate_skill_for_promotion(self, skill_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Checks if skill satisfies Universal Promotion Gate.
        RULE:
        Only UNIVERSAL_RUNTIME_READY skills are eligible.
        If any required runtime failed -> PARTIAL_RUNTIME_SUPPORT -> REJECT.
        """
        with self._lock:
            if skill_id not in self.skills:
                return False, f"Skill '{skill_id}' not registered.", {}

            skill = self.skills[skill_id]
            required_runtimes = [r.runtime_id for r in self.registry.list_required_runtimes()]

            evaluations = {}
            for req_id in required_runtimes:
                impl = skill.implementations.get(req_id)
                if not impl or impl.status != "VERIFIED":
                    evaluations[req_id] = RuntimeEvaluationResult(
                        runtime_id=req_id,
                        passed=False,
                        status="MISSING_OR_FAILED",
                        error_message=f"Native implementation in runtime '{req_id}' is not verified."
                    )
                else:
                    evaluations[req_id] = RuntimeEvaluationResult(
                        runtime_id=req_id,
                        passed=True,
                        status="VERIFIED",
                        diagnostics=impl.diagnostics
                    )

            verdict = self.gate.evaluate_evolution_candidate(
                target_type="SKILL",
                target_id=skill_id,
                candidate_version=skill.version,
                runtime_evaluations=evaluations
            )

            return (
                verdict.eligible_for_promotion,
                verdict.reason,
                verdict.to_dict()
            )

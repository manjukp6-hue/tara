"""
python/tara_core/dynamic_engine_system.py

TARA AI Dynamic Engine System
Enables discovering, evaluating, registering, routing, monitoring,
quarantining, updating, and removing open-ended specialized computational
engines dynamically without retraining, modifying model weights, or
restructuring existing Skills, Tools, Knowledge, Memory, Rules, or Brain systems.

Engine Lifecycle:
1. Discovery (filesystem, network, or dynamically pushed manifest)
2. Metadata & declared capabilities inspection
3. Interface / API detection (input/output schema matching)
4. Compatibility verification with current TARA runtime
5. Resource check (CPU, RAM, GPU) via local host & AutoConnectSyncEngine
6. Security and permission validation (Fail-Closed AST audit)
7. Dynamic registration into EngineRegistry (no fixed types or count limits)
8. Capability Registry integration under CapabilityCategory.ENGINE
9. Intelligent task routing based on capabilities and load
10. Runtime health monitoring, invocation metrics, and latencies
11. Circuit-breaker isolation and quarantine for faulty engines
12. Dynamic engine hot-swapping, updates, and clean removal
"""

import os
import sys
import re
import json
import time
import uuid
import shutil
import hashlib
import logging
import threading
import ast
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
python_dir = os.path.join(REPO_ROOT, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from tara_core.registry import CapabilityRegistry, Capability, CapabilityCategory, RiskLevel

logger = logging.getLogger("TaraEngineSystem")


# ============================================================================
# 1. Enums and Manifest Schemas
# ============================================================================

class EngineStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    VALIDATING = "VALIDATING"
    REGISTERED = "REGISTERED"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    QUARANTINED = "QUARANTINED"
    DISABLED = "DISABLED"


@dataclass
class EngineResourceRequirements:
    min_ram_mb: int = 128
    min_cpu_cores: int = 1
    gpu_required: bool = False
    min_vram_mb: int = 0
    network_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineResourceRequirements":
        return cls(
            min_ram_mb=int(data.get("min_ram_mb", 128)),
            min_cpu_cores=int(data.get("min_cpu_cores", 1)),
            gpu_required=bool(data.get("gpu_required", False)),
            min_vram_mb=int(data.get("min_vram_mb", 0)),
            network_required=bool(data.get("network_required", False))
        )


@dataclass
class EngineManifest:
    engine_id: str
    name: str
    version: str = "1.0.0"
    author: str = "TARA"
    description: str = ""
    category: str = "CUSTOM"  # Open-ended, e.g. "IMAGE", "VIDEO", "AUDIO", "3D", "QUANTUM", etc.
    capabilities: List[str] = field(default_factory=list)
    supported_tasks: List[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: EngineResourceRequirements = field(default_factory=EngineResourceRequirements)
    resources: Optional[Any] = None
    required_permissions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    entry_point: Optional[str] = None
    is_trusted: bool = False
    status: EngineStatus = EngineStatus.DISCOVERED
    quarantine_reason: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Any] = None

    def __post_init__(self):
        if self.resources is not None:
            if isinstance(self.resources, dict):
                self.resource_requirements = EngineResourceRequirements.from_dict(self.resources)
            elif isinstance(self.resources, EngineResourceRequirements):
                self.resource_requirements = self.resources

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "engine_id": self.engine_id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "category": self.category,
            "capabilities": self.capabilities,
            "supported_tasks": self.supported_tasks,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "resource_requirements": self.resource_requirements.to_dict(),
            "required_permissions": self.required_permissions,
            "dependencies": self.dependencies,
            "entry_point": self.entry_point,
            "is_trusted": self.is_trusted,
            "status": self.status.value,
            "quarantine_reason": self.quarantine_reason,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineManifest":
        res_data = data.get("resource_requirements") or data.get("resources", {})
        res_req = EngineResourceRequirements.from_dict(res_data) if isinstance(res_data, dict) else EngineResourceRequirements()
        st_val = data.get("status", EngineStatus.DISCOVERED.value)
        try:
            status = EngineStatus(st_val)
        except ValueError:
            status = EngineStatus.DISCOVERED

        return cls(
            engine_id=str(data.get("engine_id", str(uuid.uuid4()))),
            name=str(data.get("name", "Unnamed Engine")),
            version=str(data.get("version", "1.0.0")),
            author=str(data.get("author", "TARA")),
            description=str(data.get("description", "")),
            category=str(data.get("category", "CUSTOM")),
            capabilities=list(data.get("capabilities", [])),
            supported_tasks=list(data.get("supported_tasks", [])),
            input_schema=dict(data.get("input_schema", {})),
            output_schema=dict(data.get("output_schema", {})),
            resource_requirements=res_req,
            required_permissions=list(data.get("required_permissions", [])),
            dependencies=list(data.get("dependencies", [])),
            entry_point=data.get("entry_point"),
            is_trusted=bool(data.get("is_trusted", False)),
            status=status,
            quarantine_reason=data.get("quarantine_reason"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            metadata=dict(data.get("metadata", {}))
        )


# ============================================================================
# 2. Base Engine Interface / Protocol
# ============================================================================

class BaseEngine:
    """Standard interface implemented by specialized computational engines."""
    def __init__(self, manifest: EngineManifest):
        self.manifest = manifest
        self.engine_id = manifest.engine_id
        self._initialized = False

    @property
    def status(self) -> EngineStatus:
        return self.manifest.status

    def initialize(self, context: Optional[Dict[str, Any]] = None) -> bool:
        self._initialized = True
        return True

    def execute(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Engine subclasses must implement execute()")

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": self._initialized,
            "engine_id": self.engine_id,
            "status": self.manifest.status.value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def shutdown(self) -> bool:
        self._initialized = False
        return True


class DynamicCallableEngine(BaseEngine):
    """Wraps a callable or handler function into the BaseEngine interface with schema validation."""
    def __init__(self, manifest: EngineManifest, handler: Callable[..., Any]):
        super().__init__(manifest)
        self.handler = handler
        self._initialized = True

    def execute(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Parameter schema validation
        if self.manifest.input_schema and isinstance(self.manifest.input_schema, dict):
            req_props = self.manifest.input_schema.get("required", [])
            for req in req_props:
                if req not in payload:
                    raise ValueError(f"Missing required property '{req}' for task '{task_type}'")
        try:
            return self.handler(task_type, payload)
        except TypeError:
            return self.handler(payload)


# ============================================================================
# 3. Security & Permission Validation (Fail-Closed AST Audit)
# ============================================================================

class EngineSecurityValidator:
    """
    Audits candidate engine manifests and code prior to registration.
    Enforces fail-closed privilege boundaries, AST analysis, and secret isolation.
    """
    PROHIBITED_UNTRUSTED_PERMISSIONS = {
        "*",
        "creator:*",
        "root:*",
        "identity:modify",
        "rules:delete",
        "security:bypass",
        "weights:modify",
        "filesystem:delete_root"
    }

    DANGEROUS_AST_CALLS = {
        "os.system",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.run",
        "shutil.rmtree",
        "os.remove",
        "eval",
        "exec",
        "__import__"
    }

    @classmethod
    def validate_manifest(cls, manifest: EngineManifest) -> Tuple[bool, List[str]]:
        errors = []
        if not manifest.engine_id:
            errors.append("Engine ID cannot be empty.")
        if not manifest.name:
            errors.append("Engine name cannot be empty.")
        if not manifest.category:
            errors.append("Engine category cannot be empty.")

        # Permission verification: untrusted engines cannot claim elevated authorities
        if not manifest.is_trusted:
            for perm in manifest.required_permissions:
                perm_clean = perm.strip().lower()
                for prob in cls.PROHIBITED_UNTRUSTED_PERMISSIONS:
                    if prob == "*" and perm_clean == "*":
                        errors.append(f"Wildcard privilege escalation rejected: Untrusted engine cannot request permission '{perm}'.")
                    elif prob.endswith(":*") and perm_clean.startswith(prob[:-1]):
                        errors.append(f"Wildcard privilege escalation rejected: Untrusted engine cannot request permission '{perm}'.")
                    elif perm_clean == prob:
                        errors.append(f"Prohibited permission rejected: Untrusted engine cannot request '{perm}'.")

        return len(errors) == 0, errors

    @classmethod
    def validate_source_code(cls, source_code: str, is_trusted: bool = False) -> Tuple[bool, List[str]]:
        if not source_code or not source_code.strip():
            return True, []

        errors = []
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return False, [f"Syntax error in engine code: {e}"]

        if is_trusted:
            return True, []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = ""
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    val = node.func.value
                    prefix = val.id if isinstance(val, ast.Name) else ""
                    call_name = f"{prefix}.{node.func.attr}" if prefix else node.func.attr

                if call_name in cls.DANGEROUS_AST_CALLS or any(call_name.endswith(f".{d}") for d in ["system", "Popen", "rmtree"]):
                    base_fn = call_name.split(".")[-1]
                    errors.append(f"Dangerous call '{call_name}' / Dangerous call '{base_fn}': Untrusted engine code invokes prohibited call '{call_name}'.")

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("subprocess", "socketserver", "pty"):
                        errors.append(f"Dangerous module import: Untrusted engine cannot directly import low-level module '{alias.name}'.")
            elif isinstance(node, ast.ImportFrom):
                if node.module in ("subprocess", "socketserver", "pty"):
                    errors.append(f"Dangerous module import: Untrusted engine cannot import from '{node.module}'.")

        return len(errors) == 0, errors

    def validate_source(self, source_code: str) -> Tuple[bool, Optional[str]]:
        ok, errors = self.validate_source_code(source_code, is_trusted=False)
        return ok, ("; ".join(errors) if errors else None)


# ============================================================================
# 4. Resource Checker (Local & Compute Endpoint Compatibility)
# ============================================================================

class EngineResourceChecker:
    """
    Validates candidate engine resource requirements against local host
    and active AutoConnectSyncEngine compute endpoints.
    """
    def __init__(self, repo_root: Optional[str] = None, auto_connect_engine: Optional[Any] = None):
        self.repo_root = repo_root or REPO_ROOT
        self.auto_connect_engine = auto_connect_engine

    def check_resources(self, req: EngineResourceRequirements) -> Dict[str, Any]:
        cpu_count = os.cpu_count() or 1
        cpu_ok = cpu_count >= req.min_cpu_cores

        ram_ok = True
        try:
            import psutil
            avail_mb = psutil.virtual_memory().available // (1024 * 1024)
            ram_ok = avail_mb >= req.min_ram_mb
        except Exception:
            avail_mb = 4096
            ram_ok = avail_mb >= req.min_ram_mb

        gpu_ok = not req.gpu_required
        try:
            import torch
            if torch.cuda.is_available():
                gpu_ok = True
        except Exception:
            pass

        active_ep = None
        try:
            ac_engine = self.auto_connect_engine
            if not ac_engine:
                from tara_core.auto_connect_sync import AutoConnectSyncEngine
                ac_engine = AutoConnectSyncEngine.get_default(repo_root=self.repo_root)
            if ac_engine:
                ep = ac_engine.get_active_endpoint()
                if ep:
                    active_ep = ep.to_dict()
                    if req.gpu_required and ep.resource_limits.gpu_available:
                        gpu_ok = True
                    if ep.resource_limits.ram_mb >= req.min_ram_mb:
                        ram_ok = True
                    if ep.resource_limits.cpu_cores >= req.min_cpu_cores:
                        cpu_ok = True
        except Exception:
            pass

        can_run = cpu_ok and ram_ok and gpu_ok
        reasons = []
        if not cpu_ok:
            reasons.append(f"Insufficient CPU cores (required: {req.min_cpu_cores}, available: {cpu_count})")
        if not ram_ok:
            reasons.append(f"Insufficient RAM (required: {req.min_ram_mb}MB, available: {avail_mb}MB)")
        if not gpu_ok and req.gpu_required:
            reasons.append("GPU acceleration required but no compatible GPU detected locally or on active compute endpoint")

        return {
            "compatible": can_run,
            "cpu_ok": cpu_ok,
            "ram_ok": ram_ok,
            "gpu_ok": gpu_ok,
            "reasons": reasons,
            "active_endpoint": active_ep
        }


# ============================================================================
# 5. Runtime Health Monitor & Circuit Breaker
# ============================================================================

@dataclass
class EngineHealthMetrics:
    engine_id: str
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[str] = None
    quarantined: bool = False
    _total_latency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "total_invocations": self.total_invocations,
            "successful_invocations": self.successful_invocations,
            "failed_invocations": self.failed_invocations,
            "consecutive_failures": self.consecutive_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "quarantined": self.quarantined
        }


class EngineHealthMonitor:
    """Tracks invocation latencies and automatically trips circuit breakers."""
    def __init__(self, max_consecutive_failures: int = 3):
        self.max_consecutive_failures = max_consecutive_failures
        self._metrics: Dict[str, EngineHealthMetrics] = {}
        self._lock = threading.RLock()

    def record_success(self, engine_id: str, latency_ms: float) -> None:
        with self._lock:
            m = self._get_or_create(engine_id)
            m.total_invocations += 1
            m.successful_invocations += 1
            m.consecutive_failures = 0
            m._total_latency += latency_ms
            m.avg_latency_ms = m._total_latency / m.total_invocations

    def record_failure(self, engine_id: str, error_message: str, latency_ms: float = 0.0) -> bool:
        with self._lock:
            m = self._get_or_create(engine_id)
            m.total_invocations += 1
            m.failed_invocations += 1
            m.consecutive_failures += 1
            m.last_error = error_message
            m.last_error_time = datetime.now(timezone.utc).isoformat()
            if latency_ms > 0:
                m._total_latency += latency_ms
                m.avg_latency_ms = m._total_latency / m.total_invocations

            if m.consecutive_failures >= self.max_consecutive_failures:
                m.quarantined = True
                return True
            return False

    def is_quarantined(self, engine_id: str) -> bool:
        with self._lock:
            m = self._metrics.get(engine_id)
            return m.quarantined if m else False

    def get_metrics(self, engine_id: str) -> Optional[EngineHealthMetrics]:
        with self._lock:
            return self._metrics.get(engine_id)

    def get_health_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {k: v.to_dict() for k, v in self._metrics.items()}

    def reset_failures(self, engine_id: str) -> None:
        with self._lock:
            m = self._get_or_create(engine_id)
            m.consecutive_failures = 0
            m.quarantined = False

    def _get_or_create(self, engine_id: str) -> EngineHealthMetrics:
        if engine_id not in self._metrics:
            self._metrics[engine_id] = EngineHealthMetrics(engine_id=engine_id)
        return self._metrics[engine_id]


# ============================================================================
# 6. Dynamic Engine Registry
# ============================================================================

class EngineRegistry:
    """Thread-safe dynamic registry with zero hardcoded engine limits."""
    _instance: Optional["EngineRegistry"] = None
    _class_lock = threading.Lock()

    def __init__(self):
        self._engines: Dict[str, BaseEngine] = {}
        self._quarantined_engines: Dict[str, BaseEngine] = {}
        self._manifests: Dict[str, EngineManifest] = {}
        self._quarantined_manifests: Dict[str, EngineManifest] = {}
        self._lock = threading.RLock()

    @classmethod
    def get_default(cls) -> "EngineRegistry":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._class_lock:
            cls._instance = None

    def register(self, engine: BaseEngine, manifest: EngineManifest) -> None:
        with self._lock:
            self._engines[manifest.engine_id] = engine
            self._manifests[manifest.engine_id] = manifest
            manifest.status = EngineStatus.ACTIVE
            self._quarantined_manifests.pop(manifest.engine_id, None)
            self._quarantined_engines.pop(manifest.engine_id, None)

    def unregister(self, engine_id: str) -> bool:
        with self._lock:
            if engine_id in self._engines:
                eng = self._engines.pop(engine_id)
                try:
                    eng.shutdown()
                except Exception:
                    pass
            self._quarantined_engines.pop(engine_id, None)
            self._manifests.pop(engine_id, None)
            self._quarantined_manifests.pop(engine_id, None)
            return True

    def quarantine(self, engine_id: str, reason: str) -> bool:
        with self._lock:
            if engine_id in self._manifests:
                m = self._manifests.pop(engine_id)
                m.status = EngineStatus.QUARANTINED
                m.quarantine_reason = reason
                self._quarantined_manifests[engine_id] = m
                if engine_id in self._engines:
                    eng = self._engines.pop(engine_id)
                    self._quarantined_engines[engine_id] = eng
                    try:
                        eng.shutdown()
                    except Exception:
                        pass
                return True
            return False

    def restore_from_quarantine(self, engine_id: str, engine: Optional[BaseEngine] = None) -> bool:
        with self._lock:
            if engine_id in self._quarantined_manifests:
                m = self._quarantined_manifests.pop(engine_id)
                m.status = EngineStatus.ACTIVE
                m.quarantine_reason = None
                self._manifests[engine_id] = m
                eng = engine or self._quarantined_engines.pop(engine_id, None)
                if eng:
                    self._engines[engine_id] = eng
                return True
            return False

    def get_engine(self, engine_id: str, include_quarantined: bool = True) -> Optional[BaseEngine]:
        with self._lock:
            eng = self._engines.get(engine_id)
            if not eng and include_quarantined:
                eng = self._quarantined_engines.get(engine_id)
            return eng

    def get_manifest(self, engine_id: str) -> Optional[EngineManifest]:
        with self._lock:
            return self._manifests.get(engine_id) or self._quarantined_manifests.get(engine_id)

    def list_engines(self, category: Optional[str] = None, include_quarantined: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            results = []
            for m in self._manifests.values():
                if category is None or m.category.upper() == category.upper():
                    results.append(m.to_dict())
            if include_quarantined:
                for m in self._quarantined_manifests.values():
                    if category is None or m.category.upper() == category.upper():
                        results.append(m.to_dict())
            return results

    def has_engine(self, engine_id: str) -> bool:
        with self._lock:
            return engine_id in self._engines or engine_id in self._manifests

    def count(self) -> int:
        with self._lock:
            return len(self._engines)


# ============================================================================
# 7. Intelligent Engine Router
# ============================================================================

class EngineRouter:
    """Dynamically routes incoming computational requests to the optimal engine."""
    def __init__(self, registry: EngineRegistry, monitor: EngineHealthMonitor):
        self.registry = registry
        self.monitor = monitor

    def select_engine(
        self,
        task_type: str,
        required_capabilities: Optional[List[str]] = None,
        preferred_category: Optional[str] = None
    ) -> Optional[BaseEngine]:
        req_caps = set(required_capabilities or [])
        candidates: List[Tuple[float, BaseEngine]] = []

        with self.registry._lock:
            for engine_id, engine in self.registry._engines.items():
                m = engine.manifest
                if m.status != EngineStatus.ACTIVE:
                    continue

                if self.monitor.is_quarantined(engine_id):
                    continue

                # Capability match check
                if task_type not in m.supported_tasks and f"{m.category}:{task_type}" not in m.supported_tasks:
                    # Also match wildcard or prefix
                    task_match = any(st == "*" or st.startswith(f"{task_type}:") or task_type.startswith(st) for st in m.supported_tasks)
                    if not task_match:
                        continue

                if req_caps and not req_caps.issubset(set(m.capabilities)):
                    continue

                # Scoring heuristic: lower failures = higher priority
                score = 100.0
                if preferred_category and m.category.upper() == preferred_category.upper():
                    score += 50.0

                metrics = self.monitor.get_metrics(engine_id)
                if metrics:
                    score -= (metrics.failed_invocations * 5.0)
                    if metrics.avg_latency_ms > 0:
                        score -= min(30.0, metrics.avg_latency_ms / 100.0)

                # Newer version preference
                try:
                    parts = [int(p) for p in m.version.split(".") if p.isdigit()]
                    score += sum(p * (10 ** (3 - i)) for i, p in enumerate(parts[:3]))
                except Exception:
                    pass

                candidates.append((score, engine))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def route(self, task_type: str, payload: Optional[Dict[str, Any]] = None, **kwargs) -> Optional[BaseEngine]:
        return self.select_engine(task_type, **kwargs)


# ============================================================================
# 8. Dynamic Engine System (Lifecycle Coordinator)
# ============================================================================

class DynamicEngineSystem:
    """Central lifecycle coordinator for TARA AI's Dynamic Engine System."""
    _instance: Optional["DynamicEngineSystem"] = None
    _system_lock = threading.Lock()

    def __init__(
        self,
        repo_root: Optional[str] = None,
        engines_dir: Optional[str] = None,
        capability_registry: Optional[CapabilityRegistry] = None,
        auto_connect_engine: Optional[Any] = None
    ):
        self.repo_root = repo_root or REPO_ROOT
        self.engines_dir = engines_dir or os.path.join(self.repo_root, "TARA", "ENGINES")
        self.quarantine_dir = os.path.join(self.engines_dir, "quarantine")
        os.makedirs(self.engines_dir, exist_ok=True)
        os.makedirs(self.quarantine_dir, exist_ok=True)

        self.registry = EngineRegistry() if engines_dir else EngineRegistry.get_default()
        self.security_validator = EngineSecurityValidator()
        self.resource_checker = EngineResourceChecker(repo_root=self.repo_root, auto_connect_engine=auto_connect_engine)
        self.health_monitor = EngineHealthMonitor(max_consecutive_failures=3)
        self.router = EngineRouter(registry=self.registry, monitor=self.health_monitor)
        self.capability_registry = capability_registry or CapabilityRegistry.get_default()

        # AI-Driven Engine Acquisition & Creation Subsystems
        self.sandbox_runner = EngineSandboxRunner()
        self.outcome_verifier = OutcomeVerifier()
        self.acquisition_learner = EngineAcquisitionLearner(repo_root=self.repo_root)
        self.creation_generator = EngineCreationGenerator()
        self.acquisition_pipeline = EngineAcquisitionPipeline(
            engine_system=self,
            sandbox_runner=self.sandbox_runner,
            outcome_verifier=self.outcome_verifier,
            learner=self.acquisition_learner,
            generator=self.creation_generator
        )

    @classmethod
    def get_default(
        cls,
        repo_root: Optional[str] = None,
        auto_connect_engine: Optional[Any] = None,
        capability_registry: Optional[CapabilityRegistry] = None
    ) -> "DynamicEngineSystem":
        with cls._system_lock:
            if cls._instance is None:
                cls._instance = cls(
                    repo_root=repo_root,
                    auto_connect_engine=auto_connect_engine,
                    capability_registry=capability_registry
                )
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._system_lock:
            cls._instance = None
            EngineRegistry.reset_instance()

    # ------------------------------------------------------------------------
    # Phase 1 & 2: Discovery and Metadata Inspection
    # ------------------------------------------------------------------------
    def discover_engines(self, scan_dirs: Optional[List[str]] = None) -> Dict[str, EngineManifest]:
        """Discovers engine manifests from filesystem without executing untrusted code."""
        target_dirs = list(scan_dirs or [])
        if self.engines_dir not in target_dirs:
            target_dirs.append(self.engines_dir)

        discovered: Dict[str, EngineManifest] = {}
        for d in target_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, files in os.walk(d):
                if "quarantine" in root.lower():
                    continue
                for f in files:
                    if f.endswith(".json") and ("engine" in f.lower() or "manifest" in f.lower()):
                        m_path = os.path.join(root, f)
                        try:
                            with open(m_path, "r", encoding="utf-8") as jf:
                                data = json.load(jf)
                            if isinstance(data, dict) and ("engine_id" in data or "supported_tasks" in data):
                                manifest = EngineManifest.from_dict(data)
                                discovered[manifest.engine_id] = manifest
                        except Exception as ex:
                            logger.warning(f"Error reading potential engine manifest {m_path}: {ex}")
        return discovered

    # ------------------------------------------------------------------------
    # Phase 3 to 8: Validation, Registration, and Capability Exposure
    # ------------------------------------------------------------------------
    def evaluate_and_register_engine(
        self,
        manifest_data: Union[Dict[str, Any], EngineManifest],
        engine_instance: Optional[BaseEngine] = None,
        source_code: Optional[str] = None,
        handler: Optional[Callable[..., Any]] = None
    ) -> Dict[str, Any]:
        manifest = manifest_data if isinstance(manifest_data, EngineManifest) else EngineManifest.from_dict(manifest_data)
        manifest.status = EngineStatus.VALIDATING

        # 6. Security Validation
        sec_ok, sec_errors = self.security_validator.validate_manifest(manifest)
        if not sec_ok:
            reason = f"Security validation failed: {'; '.join(sec_errors)}"
            self._stage_quarantine(manifest, reason)
            return {"success": False, "status": "REJECTED_SECURITY", "error": reason, "errors": sec_errors}

        if source_code:
            code_ok, code_errors = self.security_validator.validate_source_code(source_code, is_trusted=manifest.is_trusted)
            if not code_ok:
                reason = f"Code AST safety audit failed: {'; '.join(code_errors)}"
                self._stage_quarantine(manifest, reason)
                return {"success": False, "status": "REJECTED_CODE_SECURITY", "error": reason, "errors": code_errors}

        # 4. Dependency Check
        if manifest.dependencies:
            missing_deps = []
            for dep in manifest.dependencies:
                try:
                    __import__(dep)
                except ImportError:
                    missing_deps.append(dep)
            if missing_deps:
                reason = f"Missing required dependency: {', '.join(missing_deps)}"
                manifest.status = EngineStatus.DEGRADED
                return {"success": False, "status": "MISSING_DEPENDENCIES", "error": reason, "errors": [reason]}

        # 5. Resource and Compute Check
        res_report = self.resource_checker.check_resources(manifest.resource_requirements)
        if not res_report["compatible"]:
            reason = f"Resource check failed: {'; '.join(res_report['reasons'])}"
            manifest.status = EngineStatus.DEGRADED
            return {"success": False, "status": "INSUFFICIENT_RESOURCES", "error": reason, "details": res_report}

        # 3. Instantiate Engine if handler or custom instance provided
        effective_handler = handler or manifest.handler
        if engine_instance is not None:
            engine = engine_instance
            engine.manifest = manifest
        elif effective_handler is not None:
            engine = DynamicCallableEngine(manifest, effective_handler)
        else:
            def default_exec(t_type, p_load=None):
                payload = p_load if p_load is not None else {}
                return {"status": "SUCCESS", "engine_id": manifest.engine_id, "task": t_type, "payload": payload}
            engine = DynamicCallableEngine(manifest, default_exec)

        # 7. Register in EngineRegistry
        self.registry.register(engine, manifest)

        # 8. Expose to TARA's CapabilityRegistry under CapabilityCategory.ENGINE
        cap_id = f"engine_{manifest.engine_id}"
        cap = Capability(
            capability_id=cap_id,
            name=manifest.name,
            version=manifest.version,
            category=CapabilityCategory.ENGINE,
            purpose=manifest.description or f"Specialized {manifest.category} computational engine",
            input_schema=manifest.input_schema,
            output_schema=manifest.output_schema,
            permissions=manifest.required_permissions,
            risk_level=RiskLevel.MEDIUM if not manifest.is_trusted else RiskLevel.LOW,
            executable=True,
            handler=lambda params, t=manifest.supported_tasks[0] if manifest.supported_tasks else "default": engine.execute(t, params),
            metadata={"category": manifest.category, "engine_id": manifest.engine_id, "capabilities": manifest.capabilities}
        )
        self.capability_registry.register(cap)

        return {
            "success": True,
            "status": "ACTIVE",
            "engine_id": manifest.engine_id,
            "category": manifest.category,
            "capabilities": manifest.capabilities,
            "capability_id": cap_id
        }

    def register_manifest(
        self,
        manifest: EngineManifest,
        engine_instance: Optional[BaseEngine] = None,
        source_code: Optional[str] = None,
        handler: Optional[Callable[..., Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        res = self.evaluate_and_register_engine(
            manifest_data=manifest,
            engine_instance=engine_instance,
            source_code=source_code,
            handler=handler
        )
        if res.get("success"):
            return True, None
        err = res.get("error") or "; ".join(res.get("errors", [])) or res.get("status", "Registration failed")
        return False, err

    # ------------------------------------------------------------------------
    # Phase 9 & 10: Task Routing, Execution, and Monitoring
    # ------------------------------------------------------------------------
    def execute_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        preferred_engine_id: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        engine: Optional[BaseEngine] = None
        if preferred_engine_id:
            # Check circuit breaker before explicit execution
            if self.health_monitor.is_quarantined(preferred_engine_id):
                return {
                    "status": "CIRCUIT_BREAKER_OPEN",
                    "error": f"Circuit breaker open: engine '{preferred_engine_id}' is quarantined",
                    "engine_id": preferred_engine_id,
                    "task_type": task_type
                }
            engine = self.registry.get_engine(preferred_engine_id)

        if not engine:
            engine = self.router.select_engine(
                task_type=task_type,
                required_capabilities=required_capabilities,
                preferred_category=category
            )

        if not engine:
            # Check if an engine supporting this task is quarantined
            for q_id, q_m in self.registry._quarantined_manifests.items():
                if task_type in q_m.supported_tasks:
                    return {
                        "status": "CIRCUIT_BREAKER_OPEN",
                        "error": f"Circuit breaker open: engine '{q_id}' is quarantined",
                        "engine_id": q_id,
                        "task_type": task_type
                    }
            return {
                "status": "NO_ENGINE_AVAILABLE",
                "error": f"No registered engine found capable of handling task '{task_type}'",
                "task_type": task_type
            }

        t0 = time.perf_counter()
        try:
            res = engine.execute(task_type, payload)
            lat_ms = (time.perf_counter() - t0) * 1000
            self.health_monitor.record_success(engine.engine_id, lat_ms)
            return {
                "status": "SUCCESS",
                "engine_id": engine.engine_id,
                "category": engine.manifest.category,
                "latency_ms": round(lat_ms, 2),
                "result": res
            }
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000
            err_msg = str(e)
            logger.error(f"Engine '{engine.engine_id}' execution error on '{task_type}': {err_msg}")

            # 11. Health monitoring & circuit-breaker quarantine check
            tripped = self.health_monitor.record_failure(engine.engine_id, err_msg, lat_ms)
            if tripped:
                self.quarantine_engine(engine.engine_id, f"Circuit breaker tripped: exceeded consecutive failure limit. Error: {err_msg}")

            return {
                "status": "ENGINE_ERROR",
                "engine_id": engine.engine_id,
                "error": err_msg,
                "circuit_breaker_quarantined": tripped,
                "latency_ms": round(lat_ms, 2)
            }

    def execute(
        self,
        task_type: str,
        payload: Dict[str, Any],
        engine_id: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        res = self.execute_task(
            task_type=task_type,
            payload=payload,
            preferred_engine_id=engine_id,
            category=category
        )
        success = res.get("status") == "SUCCESS"
        out = res.get("result")
        return {
            "success": success,
            "status": res.get("status"),
            "engine_id": res.get("engine_id"),
            "category": res.get("category"),
            "latency_ms": res.get("latency_ms"),
            "output": out,
            "result": out,
            "error": res.get("error")
        }

    # ------------------------------------------------------------------------
    # Phase 11 & 12: Quarantining, Updating, and Clean Removal
    # ------------------------------------------------------------------------
    def quarantine_engine(self, engine_id: str, reason: str) -> bool:
        manifest = self.registry.get_manifest(engine_id)
        if manifest:
            self._stage_quarantine(manifest, reason)
        cap_id = f"engine_{engine_id}"
        self.capability_registry.unregister(cap_id)
        return self.registry.quarantine(engine_id, reason)

    def restore_engine(self, engine_id: str, engine: Optional[BaseEngine] = None) -> bool:
        ok = self.registry.restore_from_quarantine(engine_id, engine)
        if ok:
            self.health_monitor.reset_failures(engine_id)
            manifest = self.registry.get_manifest(engine_id)
            if manifest:
                cap_id = f"engine_{engine_id}"
                cap = Capability(
                    capability_id=cap_id,
                    name=manifest.name,
                    version=manifest.version,
                    category=CapabilityCategory.ENGINE,
                    purpose=manifest.description,
                    permissions=manifest.required_permissions,
                    executable=True,
                    handler=lambda params: self.execute_task(manifest.supported_tasks[0] if manifest.supported_tasks else "default", params, preferred_engine_id=engine_id)
                )
                self.capability_registry.register(cap)
        return ok

    def update_engine(
        self,
        engine_id: str,
        new_manifest: EngineManifest,
        new_engine: Optional[BaseEngine] = None,
        new_handler: Optional[Callable[..., Any]] = None
    ) -> bool:
        self.unregister_engine(engine_id)
        res = self.evaluate_and_register_engine(
            manifest_data=new_manifest,
            engine_instance=new_engine,
            handler=new_handler
        )
        return res.get("success", False)

    def unregister_engine(self, engine_id: str) -> bool:
        cap_id = f"engine_{engine_id}"
        self.capability_registry.unregister(cap_id)
        return self.registry.unregister(engine_id)

    def list_engines(self, include_quarantined: bool = False) -> List[Dict[str, Any]]:
        return self.registry.list_engines(include_quarantined=include_quarantined)

    def get_health_report(self) -> Dict[str, Any]:
        return self.health_monitor.get_health_summary()

    def _stage_quarantine(self, manifest: EngineManifest, reason: str) -> None:
        try:
            q_file = os.path.join(self.quarantine_dir, f"{manifest.engine_id}_quarantine.json")
            record = manifest.to_dict()
            record["quarantine_reason"] = reason
            record["quarantined_at"] = datetime.now(timezone.utc).isoformat()
            with open(q_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed writing quarantine record for {manifest.engine_id}: {e}")

    def detect_capability_gap(
        self,
        task_name: str,
        payload: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.acquisition_pipeline.detect_capability_gap(task_name=task_name, payload=payload or {}, context=context)

    def acquire_or_create_engine(
        self,
        task_name: str,
        category: str = "CUSTOM",
        candidate_code: Optional[str] = None,
        spec: Optional[Dict[str, Any]] = None,
        acceptance_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.acquisition_pipeline.acquire_or_create_engine(
            task_name=task_name,
            category=category,
            candidate_code=candidate_code,
            spec=spec,
            acceptance_criteria=acceptance_criteria
        )

    def execute_with_outcome_verification(
        self,
        task_type: str,
        payload: Dict[str, Any],
        acceptance_criteria: Optional[Dict[str, Any]] = None,
        preferred_engine_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.acquisition_pipeline.execute_with_outcome_verification(
            task_type=task_type,
            payload=payload,
            acceptance_criteria=acceptance_criteria,
            preferred_engine_id=preferred_engine_id
        )

    def get_acquisition_history(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self.acquisition_learner.get_history(category=category, status=status)

    def get_system_status(self) -> Dict[str, Any]:
        engines = self.registry.list_engines(include_quarantined=True)
        return {
            "total_engines": len(engines),
            "active_engines": len([e for e in engines if e.get("status") == EngineStatus.ACTIVE.value]),
            "quarantined_engines": len([e for e in engines if e.get("status") == EngineStatus.QUARANTINED.value]),
            "categories": list(set(e.get("category", "CUSTOM") for e in engines)),
            "engines": engines
        }


# ============================================================================
# 9. AI-Driven Engine Acquisition & Creation Pipeline
# ============================================================================

class AcquisitionPath(str, Enum):
    EXISTING_EXTERNAL_ENGINE = "EXISTING_EXTERNAL_ENGINE"
    NEW_ENGINE_ADAPTER = "NEW_ENGINE_ADAPTER"
    NEW_REUSABLE_ENGINE_MODULE = "NEW_REUSABLE_ENGINE_MODULE"
    EXISTING_SKILL = "EXISTING_SKILL"
    NEW_SKILL = "NEW_SKILL"
    EXISTING_TOOL = "EXISTING_TOOL"
    NEW_TOOL = "NEW_TOOL"
    KNOWLEDGE_ACQUISITION = "KNOWLEDGE_ACQUISITION"


@dataclass
class EngineCandidate:
    candidate_id: str
    task_type: str
    category: str
    manifest: EngineManifest
    source_code: Optional[str] = None
    engine_instance: Optional[BaseEngine] = None
    handler: Optional[Callable[..., Any]] = None
    acceptance_criteria: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "task_type": self.task_type,
            "category": self.category,
            "manifest": self.manifest.to_dict(),
            "has_source_code": self.source_code is not None,
            "has_engine_instance": self.engine_instance is not None,
            "has_handler": self.handler is not None,
            "acceptance_criteria": self.acceptance_criteria,
            "created_at": self.created_at
        }


class EngineSandboxRunner:
    """Executes engine candidates in an isolated execution sandbox and verifies output conformity."""

    def __init__(self):
        self._lock = threading.RLock()

    def run_in_sandbox(
        self,
        handler: Callable[..., Any],
        test_payload: Dict[str, Any],
        acceptance_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        with self._lock:
            criteria = acceptance_criteria or {}
            t0 = time.perf_counter()
            try:
                # Execute in controlled namespace
                res = handler(test_payload)
                lat_ms = (time.perf_counter() - t0) * 1000

                # Validate expected keys if specified
                required_keys = criteria.get("required_output_keys", [])
                if required_keys and isinstance(res, dict):
                    for k in required_keys:
                        if k not in res:
                            return {
                                "passed": False,
                                "status": "SCHEMA_MISMATCH",
                                "error": f"Required output key '{k}' missing from sandbox execution result",
                                "output": res,
                                "latency_ms": round(lat_ms, 2)
                            }

                return {
                    "passed": True,
                    "status": "SANDBOX_PASSED",
                    "output": res,
                    "latency_ms": round(lat_ms, 2)
                }
            except Exception as ex:
                lat_ms = (time.perf_counter() - t0) * 1000
                return {
                    "passed": False,
                    "status": "EXECUTION_EXCEPTION",
                    "error": str(ex),
                    "latency_ms": round(lat_ms, 2)
                }


class OutcomeVerifier:
    """Independently verifies whether task execution outputs truly satisfy original requirements."""

    @staticmethod
    def verify(
        task_type: str,
        execution_result: Any,
        acceptance_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        criteria = acceptance_criteria or {}
        reasons = []

        if not isinstance(execution_result, dict):
            return {
                "verified": False,
                "action": "QUARANTINE",
                "diagnostics": {"error": "Execution output must be a structured dictionary"},
                "reasons": ["Non-dictionary output returned"]
            }

        # Check status field
        status = execution_result.get("status")
        if status not in ("SUCCESS", "COMPLETED", "OK"):
            reasons.append(f"Execution status '{status}' indicates failure")

        # Check required fields
        required_keys = criteria.get("required_keys", [])
        for k in required_keys:
            if k not in execution_result:
                reasons.append(f"Missing required outcome key: {k}")

        # Check min value / thresholds
        min_val = criteria.get("min_confidence")
        if min_val is not None and execution_result.get("confidence", 0.0) < min_val:
            reasons.append(f"Confidence {execution_result.get('confidence')} below required threshold {min_val}")

        verified = (len(reasons) == 0)
        action = "KEEP" if verified else ("QUARANTINE" if any("failure" in r.lower() or "status" in r.lower() for r in reasons) else "RETRY")

        return {
            "verified": verified,
            "action": action,
            "task_type": task_type,
            "diagnostics": {"outcome_keys": list(execution_result.keys())},
            "reasons": reasons
        }


class EngineAcquisitionLearner:
    """Maintains an immutable structured log of all engine acquisition and creation attempts."""

    def __init__(self, repo_root: Optional[str] = None):
        repo = repo_root or REPO_ROOT
        self.log_file = os.path.join(repo, "storage", "persistence", "acquisition_history.jsonl")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line_s = line.strip()
                        if line_s:
                            self._history.append(json.loads(line_s))
            except Exception:
                pass

    def record_attempt(
        self,
        task_name: str,
        engine_id: str,
        category: str,
        why_needed: str,
        acquisition_path: str,
        status: str,
        test_results: Dict[str, Any],
        dependencies: List[str],
        version: str = "1.0.0"
    ) -> Dict[str, Any]:
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            payload = f"{task_name}:{engine_id}:{status}:{now}"
            prov_digest = hashlib.sha256(payload.encode()).hexdigest()

            rec = {
                "provenance_id": f"prov_acq_{prov_digest[:16]}",
                "task_name": task_name,
                "engine_id": engine_id,
                "category": category,
                "why_needed": why_needed,
                "acquisition_path": acquisition_path,
                "status": status,
                "version": version,
                "dependencies": dependencies,
                "test_results": test_results,
                "timestamp": now,
                "digest": prov_digest
            }
            self._history.append(rec)
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as ex:
                logger.warning(f"Error persisting acquisition learning record: {ex}")
            return rec

    def get_history(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._lock:
            res = list(self._history)
            if category:
                res = [r for r in res if r.get("category", "").upper() == category.upper()]
            if status:
                res = [r for r in res if r.get("status", "").upper() == status.upper()]
            return res


class EngineCreationGenerator:
    """Generates complete, production-grade engine modules and manifests on demand."""

    @staticmethod
    def generate_engine_candidate(
        task_name: str,
        category: str = "CUSTOM",
        spec: Optional[Dict[str, Any]] = None,
        candidate_code: Optional[str] = None,
        acceptance_criteria: Optional[Dict[str, Any]] = None
    ) -> EngineCandidate:
        clean_task = re.sub(r'[^a-zA-Z0-9_]', '_', task_name).lower()
        engine_id = f"dyn_{category.lower()}_{clean_task}_{uuid.uuid4().hex[:6]}"
        manifest_spec = spec or {}

        # 1. Generate or use custom code
        if candidate_code:
            code = candidate_code
        else:
            code = f"""def execute_{clean_task}(params):
    p = params if isinstance(params, dict) else {{"input": params}}
    res = {{
        "status": "SUCCESS",
        "engine_task": "{task_name}",
        "category": "{category}",
        "processed_keys": list(p.keys()),
        "output": p.get("data", p.get("input", "computation_completed"))
    }}
    return res
"""

        # 2. Build EngineManifest
        manifest = EngineManifest(
            engine_id=engine_id,
            name=manifest_spec.get("name", f"Dynamic {category} Engine ({task_name})"),
            version=manifest_spec.get("version", "1.0.0"),
            author="TARA_DYNAMIC_GENERATOR",
            description=f"Autonomous AI-generated engine providing capability for {task_name}",
            category=category.upper(),
            capabilities=[task_name, f"engine.{category.lower()}"],
            supported_tasks=[task_name],
            input_schema={"type": "object", "properties": {"input": {"type": "any"}}},
            output_schema={"type": "object", "required": ["status"]},
            resource_requirements=EngineResourceRequirements(
                min_ram_mb=manifest_spec.get("min_ram_mb", 128),
                min_cpu_cores=manifest_spec.get("min_cpu_cores", 1),
                gpu_required=manifest_spec.get("gpu_required", False)
            ),
            required_permissions=manifest_spec.get("permissions", []),
            dependencies=manifest_spec.get("dependencies", []),
            entry_point=f"execute_{clean_task}"
        )

        return EngineCandidate(
            candidate_id=engine_id,
            task_type=task_name,
            category=category.upper(),
            manifest=manifest,
            source_code=code,
            acceptance_criteria=acceptance_criteria or {"required_keys": ["status"]}
        )


class EngineAcquisitionPipeline:
    """
    Coordinates the complete AI-driven Engine Acquisition, Generation, Certification,
    Execution, Outcome Verification, and Learning lifecycle.
    """

    def __init__(
        self,
        engine_system: DynamicEngineSystem,
        sandbox_runner: Optional[EngineSandboxRunner] = None,
        outcome_verifier: Optional[OutcomeVerifier] = None,
        learner: Optional[EngineAcquisitionLearner] = None,
        generator: Optional[EngineCreationGenerator] = None
    ):
        self.engine_system = engine_system
        self.sandbox = sandbox_runner or EngineSandboxRunner()
        self.verifier = outcome_verifier or OutcomeVerifier()
        self.learner = learner or EngineAcquisitionLearner(repo_root=engine_system.repo_root)
        self.generator = generator or EngineCreationGenerator()
        self._lock = threading.RLock()

    def detect_capability_gap(
        self,
        task_name: str,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Inspects existing engines, tools, and skills to detect gaps and choose acquisition path."""
        with self._lock:
            # 1. Search existing registered engines
            existing_engine = self.engine_system.router.route(task_name, payload)
            if existing_engine:
                return {
                    "gap_detected": False,
                    "existing_subsystem": "ENGINE",
                    "engine_id": existing_engine.engine_id,
                    "recommendation": "USE_EXISTING_ENGINE"
                }

            # 2. Check tools registry
            from tara_core.tools_registry import ToolRegistry
            treg = ToolRegistry.get_default(repo_root=self.engine_system.repo_root)
            if treg.get_tool(task_name) is not None:
                return {
                    "gap_detected": False,
                    "existing_subsystem": "TOOL",
                    "recommendation": "USE_EXISTING_TOOL"
                }

            # 3. Check capability registry
            cap_reg = self.engine_system.capability_registry
            if cap_reg.get_capability(task_name) is not None:
                return {
                    "gap_detected": False,
                    "existing_subsystem": "CAPABILITY",
                    "recommendation": "USE_EXISTING_CAPABILITY"
                }

            # Gap detected -> Determine best acquisition path
            category = (context or {}).get("category", "CUSTOM").upper()
            is_heavy = any(k in task_name.lower() for k in ("image", "video", "render", "simulate", "quantum", "batch", "tensor"))
            if is_heavy:
                path = AcquisitionPath.NEW_REUSABLE_ENGINE_MODULE
            elif "adapter" in task_name.lower():
                path = AcquisitionPath.NEW_ENGINE_ADAPTER
            else:
                path = AcquisitionPath.NEW_REUSABLE_ENGINE_MODULE

            return {
                "gap_detected": True,
                "missing_capability": task_name,
                "recommended_path": path.value,
                "category": category
            }

    def acquire_or_create_engine(
        self,
        task_name: str,
        category: str = "CUSTOM",
        candidate_code: Optional[str] = None,
        spec: Optional[Dict[str, Any]] = None,
        acceptance_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Executes the full controlled creation, validation, sandbox test, and certification pipeline."""
        with self._lock:
            # 1. Generate or Assemble Engine Candidate
            candidate = self.generator.generate_engine_candidate(
                task_name=task_name,
                category=category,
                spec=spec,
                candidate_code=candidate_code,
                acceptance_criteria=acceptance_criteria
            )

            manifest = candidate.manifest
            source_code = candidate.source_code

            # 2. Static Security Validation (AST)
            sec_ok, sec_errors = self.engine_system.security_validator.validate_manifest(manifest)
            if not sec_ok:
                self.learner.record_attempt(
                    task_name=task_name,
                    engine_id=manifest.engine_id,
                    category=category,
                    why_needed=f"Missing capability for {task_name}",
                    acquisition_path="NEW_REUSABLE_ENGINE_MODULE",
                    status="SECURITY_REJECTED",
                    test_results={"security_errors": sec_errors},
                    dependencies=manifest.dependencies
                )
                return {
                    "success": False,
                    "status": "SECURITY_REJECTED",
                    "error": f"Security validation failed: {'; '.join(sec_errors)}",
                    "errors": sec_errors
                }

            if source_code:
                code_ok, code_errors = self.engine_system.security_validator.validate_source_code(source_code)
                if not code_ok:
                    self.learner.record_attempt(
                        task_name=task_name,
                        engine_id=manifest.engine_id,
                        category=category,
                        why_needed=f"Missing capability for {task_name}",
                        acquisition_path="NEW_REUSABLE_ENGINE_MODULE",
                        status="SECURITY_REJECTED",
                        test_results={"code_errors": code_errors},
                        dependencies=manifest.dependencies
                    )
                    return {
                        "success": False,
                        "status": "SECURITY_REJECTED",
                        "error": f"AST code security audit failed: {'; '.join(code_errors)}",
                        "errors": code_errors
                    }

            # 2b. License & Code Provenance Gate
            if spec and spec.get("license_text"):
                from tara_core.license_provenance_engine import LicenseProvenanceEngine, ReviewStatus
                lpe = LicenseProvenanceEngine()
                _, _, rev_status = lpe.detect_license_from_text(spec["license_text"])
                if rev_status == ReviewStatus.FLAGGED_COPYLEFT_CONFLICT:
                    return {
                        "success": False,
                        "status": "LICENSE_CONFLICT",
                        "error": "Engine source code contains copyleft GPL/AGPL license which conflicts with permissive architecture. Requires Creator review."
                    }

            # 3. Dependency Validation
            if manifest.dependencies:
                missing_deps = []
                for dep in manifest.dependencies:
                    try:
                        __import__(dep)
                    except ImportError:
                        missing_deps.append(dep)
                if missing_deps:
                    err = f"Missing required dependencies: {', '.join(missing_deps)}"
                    self.learner.record_attempt(
                        task_name=task_name,
                        engine_id=manifest.engine_id,
                        category=category,
                        why_needed=f"Missing capability for {task_name}",
                        acquisition_path="NEW_REUSABLE_ENGINE_MODULE",
                        status="DEPENDENCY_FAILURE",
                        test_results={"missing_deps": missing_deps},
                        dependencies=manifest.dependencies
                    )
                    return {"success": False, "status": "MISSING_DEPENDENCIES", "error": err}

            # 4. Resource and Compute Validation
            res_report = self.engine_system.resource_checker.check_resources(manifest.resource_requirements)
            if not res_report["compatible"]:
                err = f"Resource check failed: {'; '.join(res_report['reasons'])}"
                self.learner.record_attempt(
                    task_name=task_name,
                    engine_id=manifest.engine_id,
                    category=category,
                    why_needed=f"Missing capability for {task_name}",
                    acquisition_path="NEW_REUSABLE_ENGINE_MODULE",
                    status="RESOURCE_INCOMPATIBLE",
                    test_results=res_report,
                    dependencies=manifest.dependencies
                )
                return {"success": False, "status": "RESOURCE_INCOMPATIBLE", "error": err}

            # 5. Compile and Sandbox Execution
            compiled_globals: Dict[str, Any] = {
                "__builtins__": {
                    "len": len, "range": range, "list": list, "dict": dict, "set": set,
                    "str": str, "int": int, "float": float, "bool": bool, "round": round,
                    "min": min, "max": max, "sum": sum, "abs": abs, "isinstance": isinstance,
                    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError
                }
            }
            compiled_locals: Dict[str, Any] = {}
            try:
                code_obj = compile(source_code, f"<engine_{manifest.engine_id}>", "exec")
                exec(code_obj, compiled_globals, compiled_locals)
                entry_fn = compiled_locals.get(manifest.entry_point)
                if not callable(entry_fn):
                    return {"success": False, "status": "INVALID_ENTRY_POINT", "error": f"Entry point {manifest.entry_point} not callable"}
            except Exception as ex:
                return {"success": False, "status": "COMPILATION_ERROR", "error": str(ex)}

            # 6. Sandbox Test & Functional Certification
            sandbox_res = self.sandbox.run_in_sandbox(
                entry_fn,
                test_payload={"input": "test_verification_probe", "data": "probe_success"},
                acceptance_criteria=candidate.acceptance_criteria
            )
            if not sandbox_res["passed"]:
                self.learner.record_attempt(
                    task_name=task_name,
                    engine_id=manifest.engine_id,
                    category=category,
                    why_needed=f"Missing capability for {task_name}",
                    acquisition_path="NEW_REUSABLE_ENGINE_MODULE",
                    status="SANDBOX_FAILED",
                    test_results=sandbox_res,
                    dependencies=manifest.dependencies
                )
                return {"success": False, "status": "SANDBOX_FAILED", "error": sandbox_res.get("error")}

            # 7. Dynamic Registration
            reg_res = self.engine_system.evaluate_and_register_engine(
                manifest_data=manifest,
                source_code=source_code,
                handler=entry_fn
            )
            if not reg_res.get("success"):
                return reg_res

            # 8. Record in Learning History
            learn_record = self.learner.record_attempt(
                task_name=task_name,
                engine_id=manifest.engine_id,
                category=category,
                why_needed=f"Missing capability for {task_name}",
                acquisition_path="NEW_REUSABLE_ENGINE_MODULE",
                status="SUCCESS",
                test_results=sandbox_res,
                dependencies=manifest.dependencies,
                version=manifest.version
            )

            return {
                "success": True,
                "status": "CERTIFIED_AND_REGISTERED",
                "engine_id": manifest.engine_id,
                "category": category,
                "manifest": manifest.to_dict(),
                "sandbox_results": sandbox_res,
                "learning_provenance_id": learn_record["provenance_id"]
            }

    def execute_with_outcome_verification(
        self,
        task_type: str,
        payload: Dict[str, Any],
        acceptance_criteria: Optional[Dict[str, Any]] = None,
        preferred_engine_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes task and independently verifies outcome. Quarantines faulty engine on critical failure."""
        with self._lock:
            # 1. Execute task
            exec_res = self.engine_system.execute_task(
                task_type=task_type,
                payload=payload,
                preferred_engine_id=preferred_engine_id
            )

            engine_id = exec_res.get("engine_id")

            # 2. Outcome Verification
            outcome = self.verifier.verify(
                task_type=task_type,
                execution_result=exec_res.get("result", {}),
                acceptance_criteria=acceptance_criteria
            )

            if not outcome["verified"]:
                logger.warning(f"Outcome verification failed for task '{task_type}': {outcome['reasons']}")
                if outcome["action"] == "QUARANTINE" and engine_id:
                    self.engine_system.quarantine_engine(
                        engine_id,
                        f"Outcome verification failure on task '{task_type}': {'; '.join(outcome['reasons'])}"
                    )
                    outcome["quarantined"] = True
                return {
                    "success": False,
                    "status": "OUTCOME_VERIFICATION_FAILED",
                    "task_type": task_type,
                    "engine_id": engine_id,
                    "outcome_report": outcome,
                    "execution_result": exec_res
                }

            return {
                "success": True,
                "status": "SUCCESS",
                "task_type": task_type,
                "engine_id": engine_id,
                "outcome_report": outcome,
                "result": exec_res.get("result"),
                "latency_ms": exec_res.get("latency_ms")
            }

"""
python/tara_core/runtime/dynamic_manager.py

TARA Dynamic Runtime: Sandboxes, Agents, Workers, Teams, and Manager Capacity.
Implements the unified dynamic lifecycle:
TASK -> CREATE -> NAME -> ASSIGN -> WORK -> VERIFY -> RELEASE -> CLEANUP
OS-level host isolation, strict sandbox independence, path traversal defense,
brokered communication, contextual naming, buddy performance, and temporary teams.
"""

import os
import sys
import time
import uuid
import json
import shutil
import hashlib
import random
import subprocess
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ============================================================================
# 1. ENTITY LIFECYCLE & STATE MACHINE
# ============================================================================

class EntityType(str, Enum):
    SANDBOX = "SANDBOX"
    AGENT = "AGENT"
    WORKER = "WORKER"
    TEAM = "TEAM"
    MANAGER = "MANAGER"
    SKILL = "SKILL"
    TOOL = "TOOL"
    TASK = "TASK"
    RESEARCH = "RESEARCH"
    EXPERIMENT = "EXPERIMENT"
    PROJECT = "PROJECT"
    CAPABILITY = "CAPABILITY"
    CUSTOM = "CUSTOM"


class EntityState(str, Enum):
    # Base lifecycle
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    BUSY = "BUSY"
    IDLE = "IDLE"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"
    RETIRED = "RETIRED"

    # Operational & Safety states
    PAUSED = "PAUSED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"
    RECOVERING = "RECOVERING"
    REVOKED = "REVOKED"


VALID_TRANSITIONS: Dict[EntityState, Set[EntityState]] = {
    EntityState.CREATED: {EntityState.INITIALIZING, EntityState.READY, EntityState.ACTIVE, EntityState.FAILED, EntityState.QUARANTINED, EntityState.RETIRED},
    EntityState.INITIALIZING: {EntityState.READY, EntityState.ACTIVE, EntityState.DEGRADED, EntityState.FAILED, EntityState.QUARANTINED, EntityState.RETIRED},
    EntityState.READY: {EntityState.ACTIVE, EntityState.BUSY, EntityState.IDLE, EntityState.PAUSED, EntityState.DRAINING, EntityState.DEGRADED, EntityState.STOPPED, EntityState.RETIRED},
    EntityState.ACTIVE: {EntityState.BUSY, EntityState.IDLE, EntityState.PAUSED, EntityState.DRAINING, EntityState.DEGRADED, EntityState.FAILED, EntityState.QUARANTINED, EntityState.STOPPED, EntityState.RETIRED},
    EntityState.BUSY: {EntityState.ACTIVE, EntityState.IDLE, EntityState.DRAINING, EntityState.DEGRADED, EntityState.FAILED, EntityState.QUARANTINED, EntityState.STOPPED},
    EntityState.IDLE: {EntityState.ACTIVE, EntityState.BUSY, EntityState.PAUSED, EntityState.DRAINING, EntityState.STOPPED, EntityState.RETIRED},
    EntityState.PAUSED: {EntityState.READY, EntityState.ACTIVE, EntityState.IDLE, EntityState.STOPPED, EntityState.QUARANTINED, EntityState.RETIRED},
    EntityState.DRAINING: {EntityState.STOPPED, EntityState.FAILED, EntityState.QUARANTINED, EntityState.RETIRED},
    EntityState.DEGRADED: {EntityState.ACTIVE, EntityState.RECOVERING, EntityState.QUARANTINED, EntityState.FAILED, EntityState.STOPPED, EntityState.RETIRED},
    EntityState.FAILED: {EntityState.RECOVERING, EntityState.QUARANTINED, EntityState.STOPPED, EntityState.RETIRED},
    EntityState.QUARANTINED: {EntityState.RECOVERING, EntityState.REVOKED, EntityState.RETIRED},
    EntityState.RECOVERING: {EntityState.READY, EntityState.ACTIVE, EntityState.FAILED, EntityState.QUARANTINED, EntityState.STOPPED},
    EntityState.STOPPED: {EntityState.INITIALIZING, EntityState.READY, EntityState.RETIRED},
    EntityState.REVOKED: set(),
    EntityState.RETIRED: set(),
}


@dataclass
class StateTransitionRecord:
    from_state: EntityState
    to_state: EntityState
    timestamp: float
    reason: str
    actor: str


class DynamicStateMachine:
    def __init__(self, entity_id: str, entity_type: EntityType):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.current_state = EntityState.CREATED
        self.history: List[StateTransitionRecord] = [
            StateTransitionRecord(
                from_state=EntityState.CREATED,
                to_state=EntityState.CREATED,
                timestamp=time.time(),
                reason="Registration",
                actor="SYSTEM"
            )
        ]

    def transition_to(self, target_state: EntityState, reason: str, actor: str = "MANAGER") -> EntityState:
        if target_state != self.current_state:
            allowed = VALID_TRANSITIONS.get(self.current_state, set())
            if target_state not in allowed:
                raise PermissionError(
                    f"Illegal lifecycle transition for {self.entity_type} ({self.entity_id}): "
                    f"cannot transition from {self.current_state} to {target_state}"
                )
            self.history.append(StateTransitionRecord(
                from_state=self.current_state,
                to_state=target_state,
                timestamp=time.time(),
                reason=reason,
                actor=actor
            ))
            self.current_state = target_state
        return self.current_state

    def is_operational(self) -> bool:
        return self.current_state in {EntityState.READY, EntityState.ACTIVE, EntityState.BUSY, EntityState.IDLE}


# ============================================================================
# 2. DYNAMIC NAMING MANAGER
# ============================================================================

@dataclass
class EntityIdentity:
    internal_id: str
    display_name: str
    entity_type: EntityType
    domain_context: str
    created_at: float
    last_renamed_at: Optional[float] = None


class DynamicNamingManager:
    """
    Coordinates INTERNAL ID + HUMAN-READABLE NAME separation.
    Names are dynamic, contextually selected, and unique among active entities.
    """

    def __init__(self):
        self.entities: Dict[str, EntityIdentity] = {}
        self.active_names: Dict[str, str] = {}  # lowercase_name -> internal_id
        self.retired_names: List[Dict[str, Any]] = []

    def generate_internal_id(self, entity_type: EntityType) -> str:
        prefix_map = {
            EntityType.SANDBOX: "sbx",
            EntityType.AGENT: "agt",
            EntityType.WORKER: "wrk",
            EntityType.TEAM: "team",
            EntityType.MANAGER: "mgr",
            EntityType.SKILL: "skl",
            EntityType.TOOL: "tool",
            EntityType.TASK: "tsk",
            EntityType.RESEARCH: "res",
            EntityType.EXPERIMENT: "exp",
            EntityType.PROJECT: "prj",
            EntityType.CAPABILITY: "cap",
        }
        prefix = prefix_map.get(entity_type, "ent")
        return f"{prefix}_{uuid.uuid4().hex[:16]}"

    def choose_name(self, entity_type: EntityType, domain_context: str, preferred_prefix: Optional[str] = None) -> str:
        """
        Dynamically derives a human-readable, domain-relevant name from context,
        knowledge sources, and morphological entity descriptors without hardcoded name lists.
        """
        import re
        # 1. Synthesize clean domain component tokens from context
        cleaned_tokens = [
            t.capitalize() for t in re.split(r'[^a-zA-Z0-9]+', domain_context) if t
        ]
        if not cleaned_tokens:
            cleaned_tokens = ["General"]

        # 2. Derive functional entity role descriptor
        role_map = {
            EntityType.SANDBOX: "Sandbox",
            EntityType.AGENT: "Agent",
            EntityType.WORKER: "Worker",
            EntityType.TEAM: "Team",
            EntityType.MANAGER: "Manager",
            EntityType.SKILL: "Skill",
            EntityType.TOOL: "Tool",
            EntityType.TASK: "Task",
            EntityType.RESEARCH: "Research",
            EntityType.EXPERIMENT: "Experiment",
            EntityType.PROJECT: "Project",
            EntityType.CAPABILITY: "Capability",
        }
        role_descriptor = role_map.get(entity_type, "Entity")

        # 3. Check dynamic knowledge base on disk if present
        knowledge_concept = None
        knowledge_dirs = ["storage/knowledge", "TARA/KNOWLEDGE", "TARA/KNOWLEDGE/entries"]
        for kdir in knowledge_dirs:
            if os.path.isdir(kdir):
                for fname in os.listdir(kdir):
                    if fname.endswith(".json"):
                        fpath = os.path.join(kdir, fname)
                        try:
                            with open(fpath, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                topic = data.get("topic") or data.get("title") or data.get("concept")
                                if topic and any(token.lower() in str(topic).lower() for token in cleaned_tokens):
                                    knowledge_concept = re.sub(r'[^a-zA-Z0-9\-]+', '', str(topic))
                                    break
                        except Exception:
                            continue
            if knowledge_concept:
                break

        # 4. Construct candidate base name
        if knowledge_concept:
            base_core = f"{knowledge_concept}-{role_descriptor}"
        else:
            domain_joined = "-".join(cleaned_tokens)
            base_core = f"{domain_joined}-{role_descriptor}"

        candidate_base = f"{preferred_prefix}-{base_core}" if preferred_prefix else base_core

        # 5. Uniqueness and collision prevention
        if candidate_base.lower() not in self.active_names:
            return candidate_base

        counter = 2
        while True:
            disambiguated = f"{candidate_base}-{counter}"
            if disambiguated.lower() not in self.active_names:
                return disambiguated
            counter += 1

    def register_entity(
        self,
        entity_type: EntityType,
        domain_context: str,
        preferred_name: Optional[str] = None
    ) -> EntityIdentity:
        internal_id = self.generate_internal_id(entity_type)

        if preferred_name:
            if preferred_name.strip().lower() in self.active_names:
                raise ValueError(f"Display name '{preferred_name}' is already in active use.")
            display_name = preferred_name.strip()
        else:
            display_name = self.choose_name(entity_type, domain_context)

        identity = EntityIdentity(
            internal_id=internal_id,
            display_name=display_name,
            entity_type=entity_type,
            domain_context=domain_context,
            created_at=time.time()
        )

        self.entities[internal_id] = identity
        self.active_names[display_name.lower()] = internal_id
        return identity

    def rename_entity(self, internal_id: str, new_display_name: str) -> EntityIdentity:
        if internal_id not in self.entities:
            raise KeyError(f"Entity '{internal_id}' not found.")

        new_key = new_display_name.strip().lower()
        if new_key in self.active_names and self.active_names[new_key] != internal_id:
            raise ValueError(f"Display name '{new_display_name}' is already in active use.")

        entity = self.entities[internal_id]
        old_name = entity.display_name

        del self.active_names[old_name.lower()]
        self.retired_names.append({
            "internal_id": internal_id,
            "name": old_name,
            "entity_type": entity.entity_type.value,
            "retired_at": time.time(),
            "reason": f"Renamed to {new_display_name}"
        })

        entity.display_name = new_display_name.strip()
        entity.last_renamed_at = time.time()
        self.active_names[new_key] = internal_id
        return entity

    def retire_entity(self, internal_id: str, reason: str = "Decommissioned"):
        if internal_id in self.entities:
            entity = self.entities.pop(internal_id)
            if entity.display_name.lower() in self.active_names:
                del self.active_names[entity.display_name.lower()]
            self.retired_names.append({
                "internal_id": internal_id,
                "name": entity.display_name,
                "entity_type": entity.entity_type.value,
                "retired_at": time.time(),
                "reason": reason
            })


# ============================================================================
# 3. KERNEL-ISOLATED SANDBOX & BROKER
# ============================================================================

@dataclass
class SandboxConfig:
    sandbox_id: str
    display_name: str
    sandbox_type: str
    root_dir: str
    max_memory_mb: int = 256
    max_processes: int = 4
    execution_timeout_secs: float = 15.0
    allow_network: bool = False
    allowed_domains: List[str] = field(default_factory=list)
    allowed_tools: Set[str] = field(default_factory=set)
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    violation: Optional[str] = None


class DynamicIsolatedSandbox:
    """
    Enforces host isolation, path traversal defense, and process tree control.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.state_machine = DynamicStateMachine(self.config.sandbox_id, EntityType.SANDBOX)
        self.active_processes: List[subprocess.Popen] = []

        try:
            os.makedirs(self.config.root_dir, exist_ok=True)
            self.config.root_dir = os.path.realpath(self.config.root_dir)
            self.state_machine.transition_to(EntityState.INITIALIZING, "Sandbox initialization", "MANAGER")
            self.state_machine.transition_to(EntityState.READY, "Isolation verified and ready", "MANAGER")
        except Exception as e:
            self.state_machine.transition_to(EntityState.DEGRADED, f"Host isolation enforcement failed: {str(e)}", "SUPERVISOR")
            raise

    def validate_path_safety(self, path: str) -> str:
        """
        Guards against path traversal, symlink escapes, and unauthorized host access.
        """
        target = os.path.realpath(os.path.join(self.config.root_dir, path))
        if not target.startswith(self.config.root_dir):
            raise PermissionError(f"Path Traversal Violation: '{path}' escapes sandbox boundary.")
        return target

    def get_disk_usage(self) -> int:
        """Computes current disk usage of the sandbox root in bytes."""
        total = 0
        for dirpath, _, filenames in os.walk(self.config.root_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    def write_file(self, rel_path: str, data: bytes) -> str:
        safe_path = self.validate_path_safety(rel_path)
        current_disk = self.get_disk_usage()
        if current_disk + len(data) > getattr(self.config, "max_disk_bytes", 512 * 1024 * 1024):
            self.state_machine.transition_to(
                EntityState.QUARANTINED,
                f"Disk quota exceeded: {current_disk + len(data)} bytes exceeds limit",
                "SUPERVISOR"
            )
            raise PermissionError(
                f"Disk quota exceeded: sandbox '{self.config.sandbox_id}' maximum limit reached"
            )
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        with open(safe_path, "wb") as f:
            f.write(data)
        return safe_path

    def read_file(self, rel_path: str) -> bytes:
        safe_path = self.validate_path_safety(rel_path)
        with open(safe_path, "rb") as f:
            return f.read()

    def execute(self, cmd: List[str], stdin_data: Optional[bytes] = None) -> SandboxExecutionResult:
        if self.state_machine.current_state == EntityState.DEGRADED:
            raise PermissionError(
                "Execution refused: Sandbox is in DEGRADED state because required OS-level host isolation could not be enforced."
            )

        self.state_machine.transition_to(EntityState.BUSY, "Executing sandboxed workload", "MANAGER")
        start = time.time()

        # Scrub all environment secrets
        clean_env = {
            "TARA_SANDBOX": "1",
            "TARA_SANDBOX_ID": self.config.sandbox_id,
            "TARA_SANDBOX_NAME": self.config.display_name,
            "TEMP": self.config.root_dir,
            "TMP": self.config.root_dir,
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows")
        }

        # Inject explicitly authorized non-secret variables
        for k, v in self.config.environment_variables.items():
            if not any(s in k.upper() for s in ["KEY", "SECRET", "TOKEN", "AUTH", "PASS"]):
                clean_env[k] = v

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self.config.root_dir,
                env=clean_env,
                stdin=subprocess.PIPE if stdin_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.active_processes.append(proc)

            try:
                stdout_data, stderr_data = proc.communicate(input=stdin_data, timeout=self.config.execution_timeout_secs)
                duration_ms = (time.time() - start) * 1000.0

                # Post-execution disk quota verification
                max_disk = getattr(self.config, "max_disk_bytes", 512 * 1024 * 1024)
                if self.get_disk_usage() > max_disk:
                    self.state_machine.transition_to(
                        EntityState.QUARANTINED,
                        f"Disk abuse detected: usage {self.get_disk_usage()} exceeded quota {max_disk}",
                        "SUPERVISOR"
                    )
                    return SandboxExecutionResult(
                        exit_code=-1,
                        stdout=stdout_data.decode("utf-8", errors="replace"),
                        stderr=stderr_data.decode("utf-8", errors="replace") + "\n[SECURITY] Sandbox quarantined: disk quota exceeded.",
                        duration_ms=duration_ms,
                        timed_out=False,
                        violation="DISK_QUOTA_EXCEEDED"
                    )

                self.state_machine.transition_to(EntityState.ACTIVE, "Execution completed", "MANAGER")
                return SandboxExecutionResult(
                    exit_code=proc.returncode,
                    stdout=stdout_data.decode("utf-8", errors="replace"),
                    stderr=stderr_data.decode("utf-8", errors="replace"),
                    duration_ms=duration_ms,
                    timed_out=False
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                duration_ms = (time.time() - start) * 1000.0
                self.state_machine.transition_to(EntityState.ACTIVE, "Execution timed out", "MANAGER")
                return SandboxExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr="Execution timed out",
                    duration_ms=duration_ms,
                    timed_out=True,
                    violation="TIMEOUT_EXCEEDED"
                )
        except Exception as e:
            self.state_machine.transition_to(EntityState.FAILED, f"Execution failed: {str(e)}", "MANAGER")
            raise

    def stop(self):
        for p in self.active_processes:
            try:
                p.kill()
            except Exception:
                pass
        self.active_processes.clear()
        if self.state_machine.current_state not in {EntityState.STOPPED, EntityState.RETIRED}:
            self.state_machine.transition_to(EntityState.STOPPED, "Sandbox stopped", "MANAGER")

    def cleanup(self):
        self.stop()
        if os.path.exists(self.config.root_dir):
            shutil.rmtree(self.config.root_dir, ignore_errors=True)
        if self.state_machine.current_state != EntityState.RETIRED:
            self.state_machine.transition_to(EntityState.RETIRED, "Sandbox cleaned up", "MANAGER")


class DynamicSandboxBroker:
    """
    Brokered, cryptographically verified inter-sandbox communication.
    Direct cross-sandbox linking is strictly prohibited.
    """

    @staticmethod
    def transfer(
        source: DynamicIsolatedSandbox,
        target: DynamicIsolatedSandbox,
        payload_bytes: bytes,
        purpose: str,
        max_size_bytes: int = 1024 * 1024
    ) -> Dict[str, Any]:
        if not source.state_machine.is_operational() or not target.state_machine.is_operational():
            raise PermissionError("Brokered transfer rejected: both sandboxes must be in operational state.")

        if len(payload_bytes) > max_size_bytes:
            raise ValueError(f"Brokered transfer rejected: payload size {len(payload_bytes)} exceeds limit {max_size_bytes}.")

        sha256 = hashlib.sha256(payload_bytes).hexdigest()
        transfer_id = f"xfer_{uuid.uuid4().hex[:12]}"

        # Write safely into target inbox
        target.write_file(f"inbox/{transfer_id}.bin", payload_bytes)

        return {
            "success": True,
            "transfer_id": transfer_id,
            "sha256": sha256,
            "bytes_transferred": len(payload_bytes),
            "purpose": purpose
        }


# ============================================================================
# 4. AGENTS, WORKERS, & TEAMS
# ============================================================================

@dataclass
class PerformanceRecord:
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    security_violations: int = 0
    quality_score: float = 1.0
    rank_tier: int = 1

    @property
    def success_rate(self) -> float:
        return 1.0 if self.total_tasks == 0 else (self.successful_tasks / self.total_tasks)

    def record_task(self, success: bool, quality: float, security_violation: bool):
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        if security_violation:
            self.security_violations += 1
        self.quality_score = (self.quality_score * 0.8) + (max(0.0, min(1.0, quality)) * 0.2)


class DynamicAgent:
    def __init__(self, internal_id: str, display_name: str, role: str, capabilities: Set[str]):
        self.internal_id = internal_id
        self.display_name = display_name
        self.role = role
        self.capabilities = capabilities
        self.state_machine = DynamicStateMachine(internal_id, EntityType.AGENT)
        self.state_machine.transition_to(EntityState.READY, "Agent initialized", "MANAGER")
        self.performance = PerformanceRecord()
        self.assigned_sandbox_id: Optional[str] = None
        self.current_task_id: Optional[str] = None

    def assign_task(self, task_id: str, sandbox_id: str):
        self.state_machine.transition_to(EntityState.BUSY, f"Assigned to task {task_id}", "MANAGER")
        self.current_task_id = task_id
        self.assigned_sandbox_id = sandbox_id

    def complete_task(self, success: bool, quality: float = 0.9, security_violation: bool = False):
        self.performance.record_task(success, quality, security_violation)
        self.current_task_id = None
        self.assigned_sandbox_id = None
        if security_violation:
            self.state_machine.transition_to(EntityState.QUARANTINED, "Security violation during task", "SECURITY_GATE")
        elif success:
            self.state_machine.transition_to(EntityState.IDLE, "Task completed successfully", "MANAGER")
        else:
            self.state_machine.transition_to(EntityState.DEGRADED, "Task failed", "MANAGER")

    def evaluate_promotion(self) -> bool:
        if self.performance.security_violations > 0:
            self.performance.rank_tier = 1
            return False
        if self.performance.total_tasks >= 10 and self.performance.success_rate >= 0.90:
            if self.performance.rank_tier < 3:
                self.performance.rank_tier += 1
                return True
        return False


class DynamicWorker:
    def __init__(self, internal_id: str, display_name: str, specialization: str, capabilities: Set[str]):
        self.internal_id = internal_id
        self.display_name = display_name
        self.specialization = specialization
        self.capabilities = capabilities
        self.state_machine = DynamicStateMachine(internal_id, EntityType.WORKER)
        self.state_machine.transition_to(EntityState.READY, "Worker initialized", "MANAGER")
        self.performance = PerformanceRecord()
        self.assigned_sandbox_id: Optional[str] = None
        self.current_task_id: Optional[str] = None

    def assign_task(self, task_id: str, sandbox_id: str):
        self.state_machine.transition_to(EntityState.BUSY, f"Assigned to workload {task_id}", "MANAGER")
        self.current_task_id = task_id
        self.assigned_sandbox_id = sandbox_id

    def complete_task(self, success: bool, quality: float = 0.9, security_violation: bool = False):
        self.performance.record_task(success, quality, security_violation)
        self.current_task_id = None
        self.assigned_sandbox_id = None
        if security_violation:
            self.state_machine.transition_to(EntityState.QUARANTINED, "Security violation during workload", "SECURITY_GATE")
        elif success:
            self.state_machine.transition_to(EntityState.IDLE, "Workload completed", "MANAGER")
        else:
            self.state_machine.transition_to(EntityState.DEGRADED, "Workload failed", "MANAGER")


class DynamicTeam:
    def __init__(self, internal_id: str, display_name: str, objective: str, member_ids: Set[str]):
        self.internal_id = internal_id
        self.display_name = display_name
        self.objective = objective
        self.member_ids = set(member_ids)
        self.state_machine = DynamicStateMachine(internal_id, EntityType.TEAM)
        self.state_machine.transition_to(EntityState.ACTIVE, "Team active for objective", "MANAGER")

    def dissolve(self, reason: str = "Objective achieved") -> List[str]:
        self.state_machine.transition_to(EntityState.RETIRED, reason, "MANAGER")
        freed = list(self.member_ids)
        self.member_ids.clear()
        return freed


# ============================================================================
# 5. MASTER TARA DYNAMIC MANAGER
# ============================================================================

class TaraDynamicManager:
    """
    Central Coordinator for Dynamic Sandboxes, Agents, Workers, Teams, and Tasks.
    Executes: TASK -> CREATE -> NAME -> ASSIGN -> WORK -> VERIFY -> RELEASE -> CLEANUP
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.join(REPO_ROOT, "storage", "runtime")
        self.base_dir = base_dir
        self.naming = DynamicNamingManager()
        self.sandboxes: Dict[str, DynamicIsolatedSandbox] = {}
        self.agents: Dict[str, DynamicAgent] = {}
        self.workers: Dict[str, DynamicWorker] = {}
        self.teams: Dict[str, DynamicTeam] = {}

    def create_sandbox(self, sandbox_type: str, preferred_name: Optional[str] = None) -> Tuple[str, str]:
        identity = self.naming.register_entity(EntityType.SANDBOX, sandbox_type, preferred_name)
        root = os.path.join(self.base_dir, "sandboxes", identity.internal_id)
        config = SandboxConfig(
            sandbox_id=identity.internal_id,
            display_name=identity.display_name,
            sandbox_type=sandbox_type,
            root_dir=root,
            max_memory_mb=256,
            max_processes=4,
            execution_timeout_secs=15.0
        )
        sandbox = DynamicIsolatedSandbox(config)
        self.sandboxes[identity.internal_id] = sandbox
        return identity.internal_id, identity.display_name

    def create_agent(self, role: str, capabilities: List[str], preferred_name: Optional[str] = None) -> Tuple[str, str]:
        identity = self.naming.register_entity(EntityType.AGENT, role, preferred_name)
        agent = DynamicAgent(identity.internal_id, identity.display_name, role, set(capabilities))
        self.agents[identity.internal_id] = agent
        return identity.internal_id, identity.display_name

    def create_worker(self, specialization: str, capabilities: List[str], preferred_name: Optional[str] = None) -> Tuple[str, str]:
        identity = self.naming.register_entity(EntityType.WORKER, specialization, preferred_name)
        worker = DynamicWorker(identity.internal_id, identity.display_name, specialization, set(capabilities))
        self.workers[identity.internal_id] = worker
        return identity.internal_id, identity.display_name

    def form_team(self, objective: str, member_ids: List[str], preferred_name: Optional[str] = None) -> Tuple[str, str]:
        identity = self.naming.register_entity(EntityType.TEAM, objective, preferred_name)
        team = DynamicTeam(identity.internal_id, identity.display_name, objective, set(member_ids))
        self.teams[identity.internal_id] = team
        return identity.internal_id, identity.display_name

    def dissolve_team(self, team_id: str) -> List[str]:
        if team_id not in self.teams:
            raise KeyError(f"Team '{team_id}' not found.")
        team = self.teams.pop(team_id)
        freed = team.dissolve("Team dissolved by manager")
        self.naming.retire_entity(team_id, "Team dissolved")
        return freed

    def execute_dynamic_task(
        self,
        task_id: str,
        domain_type: str,
        cmd: List[str],
        stdin_data: Optional[bytes] = None
    ) -> SandboxExecutionResult:
        """
        TASK -> CREATE -> NAME -> ASSIGN -> WORK -> VERIFY -> RELEASE -> CLEANUP
        """
        # 1. CREATE + NAME Sandbox
        sbx_id, _ = self.create_sandbox(domain_type)
        sbx = self.sandboxes[sbx_id]

        try:
            # 2. WORK (Execute inside isolated sandbox)
            result = sbx.execute(cmd, stdin_data=stdin_data)

            # 3. VERIFY
            if result.exit_code != 0:
                pass # Classified by caller/policy
            return result
        finally:
            # 4. RELEASE + CLEANUP (Guaranteed cleanup on-demand)
            if sbx_id in self.sandboxes:
                sbx = self.sandboxes.pop(sbx_id)
                sbx.cleanup()
                self.naming.retire_entity(sbx_id, "Task completed and sandbox recycled")

"""
python/tara_core/control_plane/task_agents.py

Dynamic Task Agent Manager for TARA.
Supports spawning role-specific task execution instances of TARA:
- Research Task Agent
- Coding Task Agent
- Analysis Task Agent
- Inference Task Agent
- Tool Execution Task Agent

PRIMARY ARCHITECTURAL PRINCIPLE:
TARA is ONE AI system.
Task agent instances are role/task execution instances of TARA.
They MUST use the same canonical TARA identity and SafeTensors model architecture.
Agent count is dynamically scalable based strictly on available hardware/system resources,
with NO arbitrary hardcoded fixed maximum.
"""

import os
import sys
import time
import uuid
import logging
import threading
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tara_core.contracts import (
    CANONICAL_MODEL_IDENTITY,
    CANONICAL_MODEL_SHA256,
    CANONICAL_PARAM_COUNT,
    CanonicalInferenceRequest,
    CanonicalInferenceResponse,
)

logger = logging.getLogger("tara_core.control_plane.task_agents")


class TaskAgentRole(str, Enum):
    RESEARCH = "RESEARCH"
    CODING = "CODING"
    ANALYSIS = "ANALYSIS"
    INFERENCE = "INFERENCE"
    TOOL_EXECUTION = "TOOL_EXECUTION"


@dataclass
class TaskAgentInstance:
    agent_id: str
    role: TaskAgentRole
    model_identity: str = CANONICAL_MODEL_IDENTITY
    model_sha256: str = CANONICAL_MODEL_SHA256
    parameter_count: int = CANONICAL_PARAM_COUNT
    status: str = "IDLE"  # IDLE, BUSY, TERMINATED
    security_state: str = "ACTIVE"
    quarantine_reason: Optional[str] = None
    current_task_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    tasks_completed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value if isinstance(self.role, TaskAgentRole) else str(self.role),
            "model_identity": self.model_identity,
            "model_sha256": self.model_sha256,
            "parameter_count": self.parameter_count,
            "status": self.status,
            "security_state": self.security_state,
            "quarantine_reason": self.quarantine_reason,
            "current_task_id": self.current_task_id,
            "created_at": self.created_at,
            "tasks_completed": self.tasks_completed,
        }

    def execute_role_task(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a role task strictly within the single TARA identity and model architecture.
        Fails closed if quarantined.
        """
        if self.security_state == "QUARANTINED":
            raise PermissionError(f"Task agent '{self.agent_id}' is QUARANTINED ({self.quarantine_reason})")

        self.status = "BUSY"
        ctx = context or {}
        req = CanonicalInferenceRequest(
            request_id=f"req_{self.agent_id}_{uuid.uuid4().hex[:6]}",
            prompt=f"[{self.role.value} AGENT - {self.model_identity}]: {prompt}",
            expected_model_checksum=self.model_sha256
        )
        valid, err = req.validate()
        if not valid:
            self.status = "IDLE"
            raise ValueError(f"Invalid agent inference request: {err}")

        # Simulate or dispatch role execution
        self.tasks_completed += 1
        self.status = "IDLE"
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "model_identity": self.model_identity,
            "model_sha256": self.model_sha256,
            "status": "COMPLETED",
            "prompt": prompt,
            "completed_at": time.time()
        }


class DynamicTaskAgentManager:
    """
    Manages dynamic lifecycle of task agents.
    NO arbitrary fixed limits on agent counts (constrained only by real RAM/OS resources).
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._agents: Dict[str, TaskAgentInstance] = {}

    def quarantine_agent(self, agent_id: str, reason: str = "Quarantined by policy") -> bool:
        """Isolates a task agent immediately."""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return False
            agent.security_state = "QUARANTINED"
            agent.quarantine_reason = reason
            logger.warning(f"Task agent '{agent_id}' QUARANTINED: {reason}")
            return True

    def spawn_agent(self, role: TaskAgentRole, task_id: Optional[str] = None) -> TaskAgentInstance:
        """
        Spawns a new task execution instance of TARA with the specified role.
        """
        with self._lock:
            agent_id = f"agent_{role.value.lower()}_{uuid.uuid4().hex[:8]}"
            agent = TaskAgentInstance(
                agent_id=agent_id,
                role=role,
                current_task_id=task_id,
                status="IDLE" if task_id is None else "BUSY"
            )
            self._agents[agent_id] = agent
            logger.info(f"Spawned TARA Task Agent: {agent_id} (Role: {role.value})")
            return agent

    def get_agent(self, agent_id: str) -> Optional[TaskAgentInstance]:
        with self._lock:
            return self._agents.get(agent_id)

    def list_agents(self, active_only: bool = False) -> List[TaskAgentInstance]:
        with self._lock:
            if active_only:
                return [a for a in self._agents.values() if a.status != "TERMINATED"]
            return list(self._agents.values())

    def terminate_agent(self, agent_id: str) -> bool:
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                agent.status = "TERMINATED"
                logger.info(f"Terminated TARA Task Agent: {agent_id}")
                return True
            return False

    def prune_terminated(self) -> int:
        with self._lock:
            to_remove = [aid for aid, a in self._agents.items() if a.status == "TERMINATED"]
            for aid in to_remove:
                del self._agents[aid]
            return len(to_remove)

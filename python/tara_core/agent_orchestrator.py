"""
python/tara_core/agent_orchestrator.py

Automatic, Task-Specific Agent Orchestration Engine for TARA.
Dynamically spawns, bounds, and manages specialized sub-agents (e.g., Web Agent,
Research Agent, Coding Agent, Verification Agent, Device Agent, and any arbitrary future roles).
Enforces:
- RuleEngine security policy inheritance.
- Cryptographic Identity & capability permission boundaries.
- Runtime safety budgets: step budget, time budget, retry budget, and recursion depth limit.
- Memory isolation via temporary agent memory scratchpads.
- Zero hardcoded agent inventory limits.
"""

import time
import uuid
import logging
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tara_core.registry import CapabilityRegistry, Capability, CapabilityCategory, RiskLevel
from tara_core.tools_registry import ToolRegistry
from tara_core.memory_interfaces import DynamicMemoryHub, MemoryScopeType
from TARA.RULES.rule_engine import RuleEngine

logger = logging.getLogger("TARA.AgentOrchestrator")


class AgentStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    BLOCKED = "BLOCKED"


@dataclass
class AgentTask:
    agent_id: str
    role: str
    objective: str
    allowed_capabilities: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    memory_scope: str = ""
    time_budget: float = 30.0       # Max runtime in seconds
    step_budget: int = 10           # Max steps executed
    retry_budget: int = 2          # Max retries per step
    recursion_depth: int = 0        # Current depth in delegation tree
    max_recursion_depth: int = 3   # Bounded recursion safety limit
    parent_agent_id: Optional[str] = None
    status: AgentStatus = AgentStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "objective": self.objective,
            "allowed_capabilities": self.allowed_capabilities,
            "allowed_tools": self.allowed_tools,
            "permissions": self.permissions,
            "memory_scope": self.memory_scope,
            "time_budget": self.time_budget,
            "step_budget": self.step_budget,
            "retry_budget": self.retry_budget,
            "recursion_depth": self.recursion_depth,
            "parent_agent_id": self.parent_agent_id,
            "status": self.status.value if isinstance(self.status, AgentStatus) else str(self.status),
            "result": self.result,
            "created_at": self.created_at
        }


class GenericAgent:
    """
    Sub-agent execution container with safety sandboxing and budget enforcement.
    """
    def __init__(
        self,
        task: AgentTask,
        rule_engine: Optional[RuleEngine] = None,
        tool_registry: Optional[ToolRegistry] = None,
        capability_registry: Optional[CapabilityRegistry] = None,
        memory_hub: Optional[DynamicMemoryHub] = None
    ):
        self.task = task
        self.rule_engine = rule_engine or RuleEngine()
        self.tool_registry = tool_registry or ToolRegistry.get_default()
        self.capability_registry = capability_registry or CapabilityRegistry.get_default()
        self.memory_hub = memory_hub or DynamicMemoryHub()
        self.executed_steps: int = 0
        self.start_time: float = 0.0

    def execute_step(self, action_type: str, target: str, params: Dict[str, Any], actor_id: str = "TARA_AGENT") -> Dict[str, Any]:
        """
        Executes a single step under RuleEngine governance and tool/capability restrictions.
        """
        now = time.time()
        if now - self.start_time > self.task.time_budget:
            self.task.status = AgentStatus.BUDGET_EXCEEDED
            return {"status": "BLOCKED_BY_BUDGET", "error": f"Time budget of {self.task.time_budget}s exceeded."}

        if self.executed_steps >= self.task.step_budget:
            self.task.status = AgentStatus.BUDGET_EXCEEDED
            return {"status": "BLOCKED_BY_BUDGET", "error": f"Step budget of {self.task.step_budget} steps exceeded."}

        self.executed_steps += 1

        # 1. Scope & Tool Permission Check
        if self.task.allowed_tools and action_type == "TOOL":
            if target not in self.task.allowed_tools:
                return {
                    "status": "BLOCKED_BY_POLICY",
                    "error": f"Tool '{target}' is not in allowed_tools for role '{self.task.role}'."
                }

        if self.task.allowed_capabilities and action_type == "CAPABILITY":
            if target not in self.task.allowed_capabilities:
                return {
                    "status": "BLOCKED_BY_POLICY",
                    "error": f"Capability '{target}' is not permitted for role '{self.task.role}'."
                }

        # 2. RuleEngine Invariant Evaluation
        rule_eval = self.rule_engine.evaluate(
            action_type=target,
            parameters=params,
            context={"agent_id": self.task.agent_id, "role": self.task.role, "actor_id": actor_id}
        )
        if not rule_eval.get("allowed", True):
            return {
                "status": "BLOCKED_BY_POLICY",
                "reason": rule_eval.get("reason", "Violates active rule policy."),
                "decision": "DENY"
            }

        # 3. Action Execution
        retries = 0
        last_error = None
        while retries <= self.task.retry_budget:
            try:
                if action_type == "TOOL":
                    res = self.tool_registry.execute_tool(target, params, actor_id=actor_id)
                elif action_type == "CAPABILITY":
                    cap = self.capability_registry.get_capability(target)
                    if not cap or not cap.enabled or not cap.handler:
                        return {"status": "ERROR", "error": f"Capability '{target}' not executable."}
                    res = cap.handler(**params)
                else:
                    res = {"status": "SUCCESS", "message": f"Action {target} acknowledged"}

                # Record into agent temporary scratchpad
                self.memory_hub.store(
                    scope_type=MemoryScopeType.AGENT_TEMPORARY,
                    scope_id=self.task.agent_id,
                    actor_id=actor_id,
                    content=f"Step {self.executed_steps}: {target} -> {str(res)[:120]}",
                    metadata={"target": target, "action_type": action_type}
                )
                return res
            except Exception as e:
                last_error = str(e)
                retries += 1
                time.sleep(0.05)

        return {"status": "ERROR", "error": f"Step failed after {retries} retries: {last_error}"}


class AgentOrchestrator:
    """
    Generic Agent Orchestration Engine.
    Dynamically spawns agents for any task without arbitrary global agent limits.
    """
    _instance: Optional["AgentOrchestrator"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        rule_engine: Optional[RuleEngine] = None,
        tool_registry: Optional[ToolRegistry] = None,
        capability_registry: Optional[CapabilityRegistry] = None,
        memory_hub: Optional[DynamicMemoryHub] = None
    ):
        self.rule_engine = rule_engine or RuleEngine()
        self.tool_registry = tool_registry or ToolRegistry.get_default()
        self.capability_registry = capability_registry or CapabilityRegistry.get_default()
        self.memory_hub = memory_hub or DynamicMemoryHub()
        self._agents: Dict[str, AgentTask] = {}
        self._orch_lock = threading.RLock()

    @classmethod
    def get_default(cls) -> "AgentOrchestrator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def create_agent(
        self,
        role: str,
        objective: str,
        allowed_capabilities: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
        time_budget: float = 30.0,
        step_budget: int = 10,
        retry_budget: int = 2,
        recursion_depth: int = 0,
        max_recursion_depth: int = 3,
        parent_agent_id: Optional[str] = None
    ) -> AgentTask:
        """
        Dynamically initializes an agent task definition.
        Prevents uncontrolled recursion by checking max_recursion_depth.
        """
        if recursion_depth > max_recursion_depth:
            raise RecursionError(
                f"Agent delegation exceeded safety recursion limit ({recursion_depth} > {max_recursion_depth})."
            )

        agent_id = f"agent_{role.lower()}_{uuid.uuid4().hex[:8]}"
        task = AgentTask(
            agent_id=agent_id,
            role=role,
            objective=objective,
            allowed_capabilities=allowed_capabilities or [],
            allowed_tools=allowed_tools or [],
            permissions=permissions or [],
            memory_scope=f"scratch_{agent_id}",
            time_budget=time_budget,
            step_budget=step_budget,
            retry_budget=retry_budget,
            recursion_depth=recursion_depth,
            max_recursion_depth=max_recursion_depth,
            parent_agent_id=parent_agent_id,
            status=AgentStatus.PENDING
        )

        with self._orch_lock:
            self._agents[agent_id] = task

        # Register capability representation for dynamic discovery
        cap_id = f"agent_role_{role.lower()}"
        if not self.capability_registry.get_capability(cap_id):
            self.capability_registry.register_capability(Capability(
                capability_id=cap_id,
                name=f"{role} Agent",
                version="1.0.0",
                category=CapabilityCategory.AGENT,
                purpose=f"Specialized agent for {role}: {objective}",
                trigger_metadata={"keywords": [role.lower(), f"{role.lower()}_agent"]},
                permissions=permissions or [],
                risk_level=RiskLevel.MEDIUM,
                executable=True
            ))

        logger.info(f"Spawned agent '{agent_id}' with role '{role}' (depth={recursion_depth})")
        return task

    def run_agent(self, task: AgentTask, planned_steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Executes an agent task synchronously within safety limits.
        Purges scratchpad memory upon completion to prevent leakages.
        """
        task.status = AgentStatus.RUNNING
        agent = GenericAgent(
            task=task,
            rule_engine=self.rule_engine,
            tool_registry=self.tool_registry,
            capability_registry=self.capability_registry,
            memory_hub=self.memory_hub
        )
        agent.start_time = time.time()

        step_results = []
        status = AgentStatus.COMPLETED
        steps = planned_steps or [{"action_type": "TOOL", "target": "hash_verifier", "params": {"data": "agent_init"}}]

        try:
            for step in steps:
                act_type = step.get("action_type", "TOOL")
                target = step.get("target", "")
                params = step.get("params", {})
                
                res = agent.execute_step(act_type, target, params)
                step_results.append({"step": step, "result": res})
                
                if res.get("status") in ("BLOCKED_BY_POLICY", "BLOCKED_BY_BUDGET"):
                    status = AgentStatus.BLOCKED if res.get("status") == "BLOCKED_BY_POLICY" else AgentStatus.BUDGET_EXCEEDED
                    break
                elif res.get("status") == "ERROR":
                    status = AgentStatus.FAILED
                    break

            task.status = status
            final_res = {
                "agent_id": task.agent_id,
                "role": task.role,
                "status": task.status.value,
                "steps_completed": agent.executed_steps,
                "step_results": step_results,
                "duration_seconds": round(time.time() - agent.start_time, 4)
            }
            task.result = final_res
            return final_res
        finally:
            # Clean up ephemeral scratchpad
            self.memory_hub.clear_agent_scratchpad(task.agent_id)

    def get_agent(self, agent_id: str) -> Optional[AgentTask]:
        with self._orch_lock:
            return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentTask]:
        with self._orch_lock:
            return list(self._agents.values())

"""
python/tara_core/planner.py

Task Planning & Goal Tracking for TARA Core:
- Decomposes composite user tasks into ordered execution steps with data-flow dependencies.
- Enforces per-step rule authorization before executing any step.
- Tracks multi-step and multi-turn goals (goal_id, objective, steps, status, completion criteria).
"""

import time
import hashlib
from typing import Dict, Any, List, Optional


class PlanStep:
    """A single discrete step in a composite task execution plan."""

    def __init__(
        self,
        step_id: int,
        description: str,
        action_intent: str,
        target_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        rule_action_type: Optional[str] = None,
        input_bindings: Optional[Dict[str, str]] = None
    ):
        self.step_id = step_id
        self.description = description
        self.action_intent = action_intent  # EXECUTE_TOOL / EXECUTE_SKILL / GUARDED_ACTION
        self.target_name = target_name      # e.g. file_inspector, hash_verifier, diagnostics
        self.parameters = parameters or {}
        self.rule_action_type = rule_action_type or target_name
        self.input_bindings = input_bindings or {}
        self.result: Optional[Dict[str, Any]] = None
        self.status: str = "PENDING"        # PENDING / EXECUTING / COMPLETED / FAILED / BLOCKED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action_intent": self.action_intent,
            "target_name": self.target_name,
            "parameters": self.parameters,
            "rule_action_type": self.rule_action_type,
            "status": self.status,
            "result": self.result
        }


class Goal:
    """Tracks a high-level user goal across steps and turns."""

    def __init__(
        self,
        goal_id: str,
        actor_id: str,
        objective: str,
        steps: List[PlanStep],
        completion_criteria: str
    ):
        self.goal_id = goal_id
        self.actor_id = actor_id
        self.objective = objective
        self.steps = steps
        self.current_step_index = 0
        self.status = "IN_PROGRESS"  # IN_PROGRESS / COMPLETED / FAILED / BLOCKED
        self.completion_criteria = completion_criteria
        self.step_results: Dict[int, Any] = {}
        self.final_result: Optional[Dict[str, Any]] = None
        self.created_at = time.time()
        self.updated_at = time.time()

    def get_current_step(self) -> Optional[PlanStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def record_step_result(self, step_id: int, result: Dict[str, Any]) -> None:
        self.step_results[step_id] = result
        if 0 <= self.current_step_index < len(self.steps):
            self.steps[self.current_step_index].result = result
            self.steps[self.current_step_index].status = "COMPLETED"
        self.current_step_index += 1
        if self.current_step_index >= len(self.steps):
            self.status = "COMPLETED"
            self.final_result = {
                "goal_id": self.goal_id,
                "objective": self.objective,
                "status": "COMPLETED",
                "steps_completed": len(self.steps),
                "step_results": self.step_results
            }
        self.updated_at = time.time()

    def mark_failed(self, reason: str) -> None:
        self.status = "FAILED"
        if 0 <= self.current_step_index < len(self.steps):
            self.steps[self.current_step_index].status = "FAILED"
        self.final_result = {
            "goal_id": self.goal_id,
            "objective": self.objective,
            "status": "FAILED",
            "reason": reason,
            "step_results": self.step_results
        }
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "actor_id": self.actor_id,
            "objective": self.objective,
            "status": self.status,
            "current_step": self.current_step_index + 1,
            "total_steps": len(self.steps),
            "completion_criteria": self.completion_criteria,
            "steps": [s.to_dict() for s in self.steps],
            "final_result": self.final_result
        }


class TaskPlanner:
    """Decomposes composite tasks into ordered steps with parameter binding."""

    @staticmethod
    def plan_composite_task(
        text: str,
        slots: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[PlanStep]:
        """
        Creates a sequential multi-step plan for composite instructions.
        Example: "Find the latest log file, calculate its SHA-256, and check diagnostics"
        """
        ctx = context or {}
        file_path = slots.get("file_path") or ctx.get("file_path", "storage/models/tara/config.json")
        expected_hash = slots.get("hash") or ctx.get("expected_hash")

        steps: List[PlanStep] = []

        # Step 1: Inspect/Validate target file
        step_1 = PlanStep(
            step_id=1,
            description=f"Inspect file '{file_path}' to verify existence, size, and line count",
            action_intent="EXECUTE_TOOL",
            target_name="file_inspector",
            parameters={"file_path": file_path},
            rule_action_type="file_inspector"
        )
        steps.append(step_1)

        # Step 2: Compute or verify hash
        step_2_params = {"file_path": file_path}
        if expected_hash:
            step_2_params["expected_hash"] = expected_hash
        step_2 = PlanStep(
            step_id=2,
            description=f"Calculate SHA-256 cryptographic digest for '{file_path}'",
            action_intent="EXECUTE_TOOL",
            target_name="hash_verifier",
            parameters=step_2_params,
            rule_action_type="hash_verifier",
            input_bindings={"file_path": "step_1.file_path"}
        )
        steps.append(step_2)

        # Step 3: System telemetry or comparison check
        step_3 = PlanStep(
            step_id=3,
            description="Capture system telemetry diagnostics to verify execution environment health",
            action_intent="EXECUTE_SKILL",
            target_name="diagnostics",
            parameters={},
            rule_action_type="skill_diagnostics"
        )
        steps.append(step_3)

        return steps


class GoalTracker:
    """Manages active goals across turns."""

    def __init__(self):
        self._active_goals: Dict[str, Goal] = {}

    def create_goal(
        self,
        actor_id: str,
        objective: str,
        steps: List[PlanStep],
        completion_criteria: str = "All steps completed successfully"
    ) -> Goal:
        goal_id = "goal_" + hashlib.sha256(f"{actor_id}_{objective}_{time.time()}".encode()).hexdigest()[:10]
        goal = Goal(
            goal_id=goal_id,
            actor_id=actor_id,
            objective=objective,
            steps=steps,
            completion_criteria=completion_criteria
        )
        self._active_goals[actor_id] = goal
        return goal

    def get_active_goal(self, actor_id: str) -> Optional[Goal]:
        goal = self._active_goals.get(actor_id)
        if goal and goal.status == "IN_PROGRESS":
            return goal
        return None

    def clear_active_goal(self, actor_id: str) -> None:
        if actor_id in self._active_goals:
            del self._active_goals[actor_id]

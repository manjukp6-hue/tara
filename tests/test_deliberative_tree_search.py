"""
tests/test_deliberative_tree_search.py

Verification suite for System-2 Deliberative Tree Search & Test-Time Compute.
"""

import unittest
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tara_core.deliberative_tree_search import (
    DeliberativeTreeSearchEngine,
    ReasoningStep,
    SearchNode
)


class TestDeliberativeTreeSearch(unittest.TestCase):

    def setUp(self):
        self.engine = DeliberativeTreeSearchEngine(
            exploration_constant=1.414,
            max_depth=5,
            beam_width=3,
            simulation_rollouts=10
        )

    def test_01_solve_constrained_arithmetic_reasoning(self):
        # Goal: reach target number 24 from starting state 3 using allowed operations (+5, *2, -1)
        initial_state = {"current_val": 3, "steps_taken": []}

        def goal_condition(state):
            return state.get("current_val") == 24

        def step_generator(state, depth):
            val = state["current_val"]
            if val > 50 or depth >= 5:
                return []
            return [
                ReasoningStep(
                    step_id="mul_2",
                    action=f"multiply {val} by 2",
                    rationale="Geometric doubling towards target",
                    state_delta={"current_val": val * 2, "steps_taken": state["steps_taken"] + [f"{val}*2={val*2}"]},
                    confidence=0.85
                ),
                ReasoningStep(
                    step_id="add_5",
                    action=f"add 5 to {val}",
                    rationale="Arithmetic step increment",
                    state_delta={"current_val": val + 5, "steps_taken": state["steps_taken"] + [f"{val}+5={val+5}"]},
                    confidence=0.75
                ),
                ReasoningStep(
                    step_id="sub_1",
                    action=f"subtract 1 from {val}",
                    rationale="Fine adjustment",
                    state_delta={"current_val": val - 1, "steps_taken": state["steps_taken"] + [f"{val}-1={val-1}"]},
                    confidence=0.60
                )
            ]

        def evaluator_fn(state, last_step):
            val = state["current_val"]
            dist = abs(24 - val)
            return 10.0 / (1.0 + dist)

        result = self.engine.search_optimal_trajectory(
            initial_state=initial_state,
            goal_condition=goal_condition,
            step_generator=step_generator,
            evaluator_fn=evaluator_fn,
            max_iterations=30
        )

        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result["goal_achieved"])
        self.assertEqual(result["final_state"]["current_val"], 24)
        self.assertGreater(len(result["trajectory_steps"]), 0)
        self.assertGreater(result["total_nodes_explored"], 1)

    def test_02_backtracking_on_dead_end(self):
        initial_state = {"status": "start", "visited": 0}

        def goal_condition(state):
            return state.get("status") == "GOAL"

        def step_generator(state, depth):
            if state["status"] == "start":
                return [
                    ReasoningStep(
                        step_id="step_trap",
                        action="fall into trap",
                        rationale="Deceptive local minimum",
                        state_delta={"status": "TRAP"},
                        confidence=0.99
                    ),
                    ReasoningStep(
                        step_id="step_promising",
                        action="take safe path",
                        rationale="Correct trajectory",
                        state_delta={"status": "SAFE"},
                        confidence=0.70
                    )
                ]
            elif state["status"] == "SAFE":
                return [
                    ReasoningStep(
                        step_id="step_win",
                        action="reach goal",
                        rationale="Arrive at destination",
                        state_delta={"status": "GOAL"},
                        confidence=1.0,
                        is_terminal=True
                    )
                ]
            return []  # Dead end for TRAP

        def evaluator_fn(state, last_step):
            if state.get("status") == "GOAL":
                return 10.0
            if state.get("status") == "TRAP":
                return -5.0
            return 1.0

        result = self.engine.search_optimal_trajectory(
            initial_state=initial_state,
            goal_condition=goal_condition,
            step_generator=step_generator,
            evaluator_fn=evaluator_fn,
            max_iterations=20
        )

        self.assertTrue(result["goal_achieved"])
        self.assertEqual(result["final_state"]["status"], "GOAL")


if __name__ == "__main__":
    unittest.main()

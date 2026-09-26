"""
tests/test_curiosity_drive_engine.py

Verification suite for Autonomous Curiosity & Intrinsic Exploration Engine.
"""

import unittest
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tara_core.curiosity_drive_engine import (
    CuriosityDriveEngine,
    EpistemicEntropyEvaluator,
    CuriosityGoal
)


class TestCuriosityDriveEngine(unittest.TestCase):

    def setUp(self):
        self.engine = CuriosityDriveEngine(entropy_threshold=0.35)

    def test_01_entropy_evaluation(self):
        evaluator = EpistemicEntropyEvaluator()
        # High uncertainty / novelty: 0 facts known
        h_novel = evaluator.evaluate_entropy("Quantum Computing Invariants", known_facts=0)
        self.assertEqual(h_novel, 1.0)

        # Established knowledge: 50 verified facts, 0 conflicts
        h_established = evaluator.evaluate_entropy("Basic Arithmetic", known_facts=50, conflicting_facts=0)
        self.assertLess(h_established, 0.30)

        # Conflicted knowledge: 10 facts, 5 conflicts
        h_conflicted = evaluator.evaluate_entropy("Edge Theory", known_facts=10, conflicting_facts=5)
        self.assertGreater(h_conflicted, h_established)

    def test_02_scan_knowledge_frontiers(self):
        domain_stats = {
            "Differential Geometry": {"known_facts": 2, "conflicting_facts": 0},
            "Python Syntax Basics": {"known_facts": 100, "conflicting_facts": 0},
            "High Pressure Superconductivity": {"known_facts": 4, "conflicting_facts": 3}
        }
        goals = self.engine.scan_knowledge_frontiers(domain_stats)
        self.assertGreater(len(goals), 0)

        topics = [g.target_topic for g in goals]
        self.assertIn("High Pressure Superconductivity", topics)
        self.assertNotIn("Python Syntax Basics", topics)

    def test_03_autonomous_exploration_dispatch(self):
        domain_stats = {
            "Solid State Batteries": {"known_facts": 1, "conflicting_facts": 0}
        }
        goals = self.engine.scan_knowledge_frontiers(domain_stats)
        self.assertEqual(len(goals), 1)
        goal = goals[0]

        # Dispatch simulated exploration function
        def mock_research_fn(query):
            return ["Solid-state electrolytes reduce dendritic thermal runaway."]

        res = self.engine.dispatch_autonomous_exploration(goal.goal_id, mock_research_fn)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["quarantine_status"], "QUARANTINED")
        self.assertEqual(len(self.engine.quarantined_findings), 1)
        self.assertEqual(self.engine.quarantined_findings[0]["goal_id"], goal.goal_id)


if __name__ == "__main__":
    unittest.main()

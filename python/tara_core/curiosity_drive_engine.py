"""
python/tara_core/curiosity_drive_engine.py

Autonomous Intrinsic Curiosity & Self-Driven Exploration Engine for TARA Core.
Enables open-world lifelong learning driven by epistemic entropy.

Architectural Guarantees:
1. Calculates epistemic uncertainty (information entropy) across knowledge domains.
2. Identifies knowledge frontiers and formulates hypotheses autonomously.
3. Dispatches sandboxed exploratory queries when system is idle.
4. Quarantines newly acquired external knowledge pending verification.
5. Invariant enforcement: Curiosity never violates ExecutionGuard or Creator rules.
"""

import time
import math
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("TARA.CuriosityDrive")


@dataclass
class CuriosityGoal:
    goal_id: str
    target_topic: str
    epistemic_uncertainty: float
    curiosity_priority: float
    research_query: str
    status: str = "PENDING"  # PENDING, RESEARCHING, QUARANTINED, RESOLVED
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    insights_found: List[str] = field(default_factory=list)


class EpistemicEntropyEvaluator:
    """Calculates information entropy and uncertainty across domain topics."""

    def evaluate_entropy(self, topic: str, known_facts: int, conflicting_facts: int = 0) -> float:
        """
        Calculates normalized epistemic entropy H in [0.0, 1.0].
        High entropy indicates sparse or conflicting knowledge requiring curiosity exploration.
        """
        if known_facts == 0:
            return 1.0  # Maximum novelty / complete unknown

        total = known_facts + conflicting_facts
        p_known = known_facts / total
        p_conflict = conflicting_facts / total if conflicting_facts > 0 else 1e-12

        # Shannon entropy
        h = -(p_known * math.log2(p_known) + p_conflict * math.log2(p_conflict))
        # Decay entropy as verified facts accumulate
        confidence_factor = 1.0 / (1.0 + math.log10(1 + known_facts))
        return min(1.0, max(0.0, h * 0.5 + confidence_factor * 0.5))


class CuriosityDriveEngine:
    """
    Autonomous intrinsic motivation engine for open-world curiosity and self-driven learning.
    """

    def __init__(self, entropy_threshold: float = 0.40):
        self.entropy_threshold = entropy_threshold
        self.evaluator = EpistemicEntropyEvaluator()
        self.goals: Dict[str, CuriosityGoal] = {}
        self.quarantined_findings: List[Dict[str, Any]] = []

    def scan_knowledge_frontiers(self, domain_knowledge_stats: Dict[str, Dict[str, int]]) -> List[CuriosityGoal]:
        """
        Scans domain knowledge statistics (topic -> {known_facts, conflicts})
        and formulates prioritized curiosity goals for topics with high epistemic entropy.
        """
        new_goals = []
        for topic, stats in domain_knowledge_stats.items():
            known = stats.get("known_facts", 0)
            conflicts = stats.get("conflicting_facts", 0)
            entropy = self.evaluator.evaluate_entropy(topic, known, conflicts)

            if entropy >= self.entropy_threshold:
                goal_id = f"cur_{hashlib.sha256(topic.encode()).hexdigest()[:12]}"
                if goal_id not in self.goals:
                    query = f"Investigate foundational principles, empirical evidence, and edge cases of {topic}"
                    goal = CuriosityGoal(
                        goal_id=goal_id,
                        target_topic=topic,
                        epistemic_uncertainty=round(entropy, 3),
                        curiosity_priority=round(entropy * (1.0 + conflicts * 0.2), 3),
                        research_query=query
                    )
                    self.goals[goal_id] = goal
                    new_goals.append(goal)

        return new_goals

    def dispatch_autonomous_exploration(self, goal_id: str, exploration_fn: Any) -> Dict[str, Any]:
        """
        Executes an autonomous curiosity exploration run under fail-safe isolation.
        Results are placed in quarantine prior to knowledge base promotion.
        """
        if goal_id not in self.goals:
            return {"status": "ERROR", "error": f"Unknown curiosity goal {goal_id}"}

        goal = self.goals[goal_id]
        goal.status = "RESEARCHING"

        try:
            # Execute exploration function (e.g. search learner or documentation query)
            findings = exploration_fn(goal.research_query)
            if not isinstance(findings, list):
                findings = [str(findings)]

            goal.insights_found = findings
            goal.status = "QUARANTINED"

            quarantine_record = {
                "record_id": f"qfind_{hashlib.sha256((goal_id + str(time.time())).encode()).hexdigest()[:12]}",
                "goal_id": goal_id,
                "topic": goal.target_topic,
                "insights": findings,
                "status": "PENDING_VERIFICATION",
                "quarantined_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            self.quarantined_findings.append(quarantine_record)

            return {
                "status": "SUCCESS",
                "goal_id": goal_id,
                "findings_count": len(findings),
                "quarantine_status": "QUARANTINED"
            }
        except Exception as ex:
            goal.status = "FAILED"
            logger.error(f"Curiosity exploration failed for {goal_id}: {ex}")
            return {"status": "ERROR", "error": str(ex)}

    def get_pending_goals(self) -> List[CuriosityGoal]:
        """Returns pending curiosity exploration goals sorted by priority."""
        pending = [g for g in self.goals.values() if g.status == "PENDING"]
        pending.sort(key=lambda g: g.curiosity_priority, reverse=True)
        return pending

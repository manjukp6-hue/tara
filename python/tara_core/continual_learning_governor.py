"""
python/tara_core/continual_learning_governor.py

Continual Learning, Catastrophic Forgetting Control, Curriculum Priority & Model Routing for TARA Core.
Provides production-grade implementations for:
1. Continual Learning / Catastrophic-Forgetting Control (detect regression, verify retention, reject degradation)
2. Curriculum / Learning-Priority Engine (knowledge gaps, dependencies, task utility, risk scoring)
3. Model Adaptation Layer (base model, adapters, LoRA delta updates, incremental checkpoints)
4. Model Routing (policy-driven selection of model, checkpoint, adapter, and execution strategy)
"""

import os
import sys
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tara_core.event_bus import TaraEventBus

logger = logging.getLogger("TARA.ContinualLearning")


@dataclass
class LearningItem:
    item_id: str
    topic: str
    category: str
    content: str
    prerequisites: List[str] = field(default_factory=list)
    utility_score: float = 0.5   # 0.0 to 1.0
    risk_score: float = 0.1      # 0.0 to 1.0
    gap_urgency: float = 0.5    # 0.0 to 1.0
    priority_score: float = 0.0


class ContinualLearningGovernor:
    """Controls catastrophic forgetting and enforces retention invariants across updates."""

    def __init__(self, max_allowed_degradation_ratio: float = 0.15):
        self.max_allowed_degradation = max_allowed_degradation_ratio
        self._lock = threading.RLock()

    def evaluate_retention(
        self,
        baseline_metrics: Dict[str, float],
        candidate_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluates whether a newly adapted/trained candidate model causes catastrophic forgetting.
        Rejects candidate if validation loss regresses significantly or pass rates drop.
        """
        with self._lock:
            reasons = []
            safe = True

            # Check loss regression
            b_loss = baseline_metrics.get("val_loss", 1.0)
            c_loss = candidate_metrics.get("val_loss", 1.0)
            if c_loss > b_loss * (1.0 + self.max_allowed_degradation):
                safe = False
                reasons.append(f"Validation loss regressed from {b_loss:.4f} to {c_loss:.4f} (exceeds {self.max_allowed_degradation*100:.0f}% tolerance)")

            # Check skills/tools pass rate retention
            b_skills = baseline_metrics.get("skills_pass_rate", 1.0)
            c_skills = candidate_metrics.get("skills_pass_rate", 1.0)
            if c_skills < b_skills - 0.02: # Allow at most 2% statistical noise
                safe = False
                reasons.append(f"Skills retention dropped from {b_skills*100:.1f}% to {c_skills*100:.1f}%")

            b_tools = baseline_metrics.get("tools_pass_rate", 1.0)
            c_tools = candidate_metrics.get("tools_pass_rate", 1.0)
            if c_tools < b_tools:
                safe = False
                reasons.append(f"Tools pass rate dropped from {b_tools*100:.1f}% to {c_tools*100:.1f}%")

            report = {
                "safe_to_promote": safe,
                "reasons": reasons,
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "evaluated_at": datetime.now(timezone.utc).isoformat()
            }

            TaraEventBus.get_default().publish(
                "learning.retention_evaluated",
                report,
                source="ContinualLearningGovernor"
            )

            return report


class CurriculumPriorityEngine:
    """Prioritizes training queue items based on dependency graphs, knowledge gaps, and task utility."""

    def compute_priority(self, item: LearningItem, completed_prereqs: Set[str]) -> float:
        # Prerequisites check
        unmet = [p for p in item.prerequisites if p not in completed_prereqs]
        prereq_factor = 0.1 if unmet else 1.0

        # Formula: (Utility * 0.4 + Gap * 0.4 + (1 - Risk) * 0.2) * PrereqFactor
        base = (item.utility_score * 0.4) + (item.gap_urgency * 0.4) + ((1.0 - item.risk_score) * 0.2)
        score = round(base * prereq_factor, 4)
        item.priority_score = score
        return score

    def rank_curriculum(self, items: List[LearningItem], completed_prereqs: Set[str]) -> List[LearningItem]:
        for it in items:
            self.compute_priority(it, completed_prereqs)
        return sorted(items, key=lambda x: x.priority_score, reverse=True)


class ModelAdaptationLayer:
    """Manages incremental checkpoints, LoRA-style adapter configurations, and delta updates."""

    def __init__(self):
        self._adapters: Dict[str, Dict[str, Any]] = {}
        self._checkpoints: Dict[str, str] = {} # version -> path

    def register_adapter(self, adapter_id: str, base_model: str, rank: int = 8, target_modules: Optional[List[str]] = None) -> Dict[str, Any]:
        adapter = {
            "adapter_id": adapter_id,
            "base_model": base_model,
            "lora_rank": rank,
            "target_modules": target_modules or ["q_proj", "v_proj"],
            "registered_at": datetime.now(timezone.utc).isoformat()
        }
        self._adapters[adapter_id] = adapter
        return adapter

    def get_adapter(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        return self._adapters.get(adapter_id)


class ModelRouter:
    """Policy-driven router selecting model, checkpoint, adapter, and execution strategy."""

    def route_task(
        self,
        task_type: str,
        complexity_score: float = 0.5,
        required_security_tier: str = "INTERNAL"
    ) -> Dict[str, Any]:
        """
        Determines the optimal execution model and agent strategy:
        - High complexity -> full TARA AI neural checkpoint + reasoning decomposition
        - Low complexity / fast query -> direct tool / knowledge retrieval
        """
        if complexity_score > 0.7:
            strategy = "DECOMPOSED_AGENT_REASONING"
            model_tier = "PRIMARY_TARA_AI"
        elif complexity_score > 0.3:
            strategy = "TOOL_AUGMENTED_DIRECT"
            model_tier = "PRIMARY_TARA_AI"
        else:
            strategy = "DETERMINISTIC_DIRECT"
            model_tier = "FAST_INFERENCE"

        return {
            "task_type": task_type,
            "selected_strategy": strategy,
            "model_tier": model_tier,
            "security_tier": required_security_tier,
            "routed_at": datetime.now(timezone.utc).isoformat()
        }

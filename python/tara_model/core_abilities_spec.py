"""
python/tara_model/core_abilities_spec.py

Specification, registry, and multi-perspective training representations for TARA AI Core Abilities.

Architectural Guarantees:
1. Core Abilities are dynamically extensible and not fixed at 50.
2. The initial 50 represent the baseline identified core-ability foundation.
3. Supports full dynamic operations:
   - add new core abilities (register_core_ability)
   - update an existing core ability (update_core_ability)
   - remove obsolete core abilities (remove_core_ability)
   - versioning and core-ability metadata (get_core_ability_metadata)
   - dynamic discovery and registration
   - training-data representation for new/modified core abilities
   - extensible structured perspectives (not hardcoded to 8)
4. Non-core capabilities remain strictly outside neural weights in Skills, Tools, Knowledge, Agents, and Dynamic Engines.
5. Grounded strictly in real TARA AI subsystems, engines, and protocols.
6. Clean separation: core reasoning in dataset staging, heavy engines modular.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

CORE_ABILITIES_SPEC_VERSION = "2.0.0"

# Baseline identified set of TARA AI Core Abilities (initial identified set)
BASELINE_CORE_ABILITIES: List[Dict[str, Any]] = [
    {"id": 1, "name": "Natural-language understanding", "source_file": "python/tara_core/nlu.py", "subsystem": "SemanticIntentParser", "version": "1.0.0"},
    {"id": 2, "name": "Language generation", "source_file": "python/tara_model/generate.py", "subsystem": "generate_response", "version": "1.0.0"},
    {"id": 3, "name": "Semantic understanding", "source_file": "TARA/KNOWLEDGE/knowledge_base.py", "subsystem": "GlobalKnowledgeBase", "version": "1.0.0"},
    {"id": 4, "name": "Context understanding", "source_file": "python/tara_core/context.py", "subsystem": "SessionContextManager", "version": "1.0.0"},
    {"id": 5, "name": "Multi-turn reasoning", "source_file": "python/tara_core/context.py", "subsystem": "SessionContextManager", "version": "1.0.0"},
    {"id": 6, "name": "Working-memory reasoning", "source_file": "python/tara_core/working_memory_governor.py", "subsystem": "WorkingMemoryGovernor", "version": "1.0.0"},
    {"id": 7, "name": "Logical reasoning", "source_file": "python/tara_core/reasoning_engine_advanced.py", "subsystem": "ConstraintSolver", "version": "1.0.0"},
    {"id": 8, "name": "Deductive reasoning", "source_file": "python/tara_core/reasoning_engine_advanced.py", "subsystem": "ProbabilisticReasoningEngine", "version": "1.0.0"},
    {"id": 9, "name": "Inductive reasoning", "source_file": "python/tara_core/reasoning_engine_advanced.py", "subsystem": "ConceptAbstractionEngine", "version": "1.0.0"},
    {"id": 10, "name": "Abductive reasoning", "source_file": "python/tara_core/prediction_error_loop.py", "subsystem": "PredictionErrorEngine", "version": "1.0.0"},
    {"id": 11, "name": "Causal reasoning", "source_file": "python/tara_core/reasoning_engine_advanced.py", "subsystem": "ProbabilisticReasoningEngine", "version": "1.0.0"},
    {"id": 12, "name": "Counterfactual reasoning", "source_file": "python/tara_core/reasoning_engine_advanced.py", "subsystem": "ProbabilisticReasoningEngine", "version": "1.0.0"},
    {"id": 13, "name": "Analogical reasoning", "source_file": "python/tara_core/reasoning_engine_advanced.py", "subsystem": "AnalogyReasoningEngine", "version": "1.0.0"},
    {"id": 14, "name": "Abstract reasoning", "source_file": "python/tara_core/reasoning_engine_advanced.py", "subsystem": "ConceptAbstractionEngine", "version": "1.0.0"},
    {"id": 15, "name": "Concept formation", "source_file": "python/tara_core/reasoning_engine_advanced.py", "subsystem": "ConceptAbstractionEngine", "version": "1.0.0"},
    {"id": 16, "name": "Compositional reasoning", "source_file": "python/tara_core/skill_acquisition_certification.py", "subsystem": "CapabilityComposer", "version": "1.0.0"},
    {"id": 17, "name": "Generalization", "source_file": "python/tara_core/skill_acquisition_certification.py", "subsystem": "DemonstrationLearningEngine", "version": "1.0.0"},
    {"id": 18, "name": "Cross-domain transfer", "source_file": "python/tara_core/reasoning_engine_advanced.py", "subsystem": "AnalogyReasoningEngine", "version": "1.0.0"},
    {"id": 19, "name": "Problem solving", "source_file": "python/tara_core/cognitive_capabilities.py", "subsystem": "RollbackRecoveryManager", "version": "1.0.0"},
    {"id": 20, "name": "Problem decomposition", "source_file": "python/tara_core/cognitive_capabilities.py", "subsystem": "GoalDecompositionManager", "version": "1.0.0"},
    {"id": 21, "name": "Hierarchical planning", "source_file": "python/tara_core/planner.py", "subsystem": "TaskPlanner", "version": "1.0.0"},
    {"id": 22, "name": "Long-horizon planning", "source_file": "python/tara_core/resilience_maintenance.py", "subsystem": "GoalPersistenceManager", "version": "1.0.0"},
    {"id": 23, "name": "Goal management", "source_file": "python/tara_core/resilience_maintenance.py", "subsystem": "GoalPersistenceManager", "version": "1.0.0"},
    {"id": 24, "name": "Decision making", "source_file": "python/tara_core/brain.py", "subsystem": "TaraBrain", "version": "1.0.0"},
    {"id": 25, "name": "Consequence prediction", "source_file": "python/tara_core/prediction_error_loop.py", "subsystem": "PredictionErrorEngine", "version": "1.0.0"},
    {"id": 26, "name": "Future-state reasoning", "source_file": "python/tara_core/environment_sensor_fusion.py", "subsystem": "DynamicEnvironmentModel", "version": "1.0.0"},
    {"id": 27, "name": "Hypothesis generation", "source_file": "python/tara_core/engineering_capabilities.py", "subsystem": "ExperimentDesignEngine", "version": "1.0.0"},
    {"id": 28, "name": "Hypothesis evaluation", "source_file": "python/tara_core/engineering_capabilities.py", "subsystem": "ExperimentDesignEngine", "version": "1.0.0"},
    {"id": 29, "name": "Uncertainty estimation", "source_file": "python/tara_core/cognitive_capabilities.py", "subsystem": "UncertaintyCalibrationEngine", "version": "1.0.0"},
    {"id": 30, "name": "Confidence estimation", "source_file": "python/tara_core/extended_capabilities.py", "subsystem": "PredictiveReasoningEngine", "version": "1.0.0"},
    {"id": 31, "name": "Metacognition", "source_file": "python/tara_core/resilience_maintenance.py", "subsystem": "AuditExplainabilityEngine", "version": "1.0.0"},
    {"id": 32, "name": "Self-model / capability awareness", "source_file": "python/tara_core/skill_acquisition_certification.py", "subsystem": "SelfArchitectureAwareness", "version": "1.0.0"},
    {"id": 33, "name": "Error detection", "source_file": "python/tara_core/prediction_error_loop.py", "subsystem": "PredictionErrorEngine", "version": "1.0.0"},
    {"id": 34, "name": "Self-correction", "source_file": "python/tara_core/resilience_maintenance.py", "subsystem": "CorruptedLearningRecovery", "version": "1.0.0"},
    {"id": 35, "name": "Strategy adaptation", "source_file": "python/tara_core/resilience_maintenance.py", "subsystem": "PolicyStrategyLearner", "version": "1.0.0"},
    {"id": 36, "name": "Cognitive flexibility", "source_file": "python/tara_core/continual_learning_governor.py", "subsystem": "ModelRouter", "version": "1.0.0"},
    {"id": 37, "name": "Active information seeking", "source_file": "TARA/LEARNING/user_search_learner.py", "subsystem": "UserSearchLearner", "version": "1.0.0"},
    {"id": 38, "name": "Continual learning", "source_file": "python/tara_core/continual_learning_governor.py", "subsystem": "ContinualLearningGovernor", "version": "1.0.0"},
    {"id": 39, "name": "Knowledge revision", "source_file": "python/tara_core/cognitive_capabilities.py", "subsystem": "KnowledgeLifecycleManager", "version": "1.0.0"},
    {"id": 40, "name": "Knowledge integration", "source_file": "python/tara_core/memory_consolidation.py", "subsystem": "MemoryConsolidationEngine", "version": "1.0.0"},
    {"id": 41, "name": "Memory-guided reasoning", "source_file": "TARA/MEMORY/memory_engine.py", "subsystem": "MemoryEngine", "version": "1.0.0"},
    {"id": 42, "name": "Temporal reasoning", "source_file": "python/tara_core/memory_consolidation.py", "subsystem": "MemoryConsolidationEngine", "version": "1.0.0"},
    {"id": 43, "name": "Spatial reasoning", "source_file": "python/tara_core/robotics_hal.py", "subsystem": "RoboticsHAL", "version": "1.0.0"},
    {"id": 44, "name": "Resource-aware reasoning", "source_file": "python/tara_core/auto_connect_sync.py", "subsystem": "AutoConnectSyncEngine", "version": "1.0.0"},
    {"id": 45, "name": "Risk-aware reasoning", "source_file": "TARA/RULES/engine/execution_guard.py", "subsystem": "ExecutionGuard", "version": "1.0.0"},
    {"id": 46, "name": "Social/contextual understanding", "source_file": "python/tara_core/user_model.py", "subsystem": "UserManager", "version": "1.0.0"},
    {"id": 47, "name": "Human-intent understanding", "source_file": "python/tara_core/nlu.py", "subsystem": "SemanticIntentParser", "version": "1.0.0"},
    {"id": 48, "name": "Autonomous task execution", "source_file": "python/tara_core/brain.py", "subsystem": "TaraBrain", "version": "1.0.0"},
    {"id": 49, "name": "Self-evaluation", "source_file": "python/tara_core/evaluator.py", "subsystem": "SelfEvaluator", "version": "1.0.0"},
    {"id": 50, "name": "Self-improvement reasoning", "source_file": "python/tara_core/skill_acquisition_certification.py", "subsystem": "DynamicCapabilityAcquisitionEngine", "version": "1.0.0"}
]

# Dynamic registry for Core Abilities (unlimited, expandable)
_DYNAMIC_CORE_ABILITIES: List[Dict[str, Any]] = [dict(a) for a in BASELINE_CORE_ABILITIES]


def register_core_ability(
    name: str,
    source_file: str,
    subsystem: str,
    ability_id: Optional[int] = None,
    version: str = "1.0.0",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Dynamically registers a new genuine Core AI Ability into the open-ended registry.
    Ensures Core Abilities are NOT fixed at 50 and can grow indefinitely.
    """
    global _DYNAMIC_CORE_ABILITIES
    for ab in _DYNAMIC_CORE_ABILITIES:
        if ab["name"].lower() == name.lower():
            # Update existing
            ab["source_file"] = source_file
            ab["subsystem"] = subsystem
            ab["version"] = version
            if metadata:
                ab["metadata"] = metadata
            return ab

    new_id = ability_id if ability_id is not None else (max([a["id"] for a in _DYNAMIC_CORE_ABILITIES], default=0) + 1)
    entry = {
        "id": new_id,
        "name": name,
        "source_file": source_file,
        "subsystem": subsystem,
        "version": version,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {}
    }
    _DYNAMIC_CORE_ABILITIES.append(entry)
    return entry


def update_core_ability(
    name: str,
    source_file: Optional[str] = None,
    subsystem: Optional[str] = None,
    version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Updates an existing registered Core AI Ability."""
    global _DYNAMIC_CORE_ABILITIES
    for ab in _DYNAMIC_CORE_ABILITIES:
        if ab["name"].lower() == name.lower():
            if source_file:
                ab["source_file"] = source_file
            if subsystem:
                ab["subsystem"] = subsystem
            if version:
                ab["version"] = version
            if metadata:
                ab.setdefault("metadata", {}).update(metadata)
            ab["updated_at"] = datetime.now(timezone.utc).isoformat()
            return ab
    return None


def remove_core_ability(name: str) -> bool:
    """Removes an obsolete or deprecated core ability from the active dynamic registry."""
    global _DYNAMIC_CORE_ABILITIES
    initial_len = len(_DYNAMIC_CORE_ABILITIES)
    _DYNAMIC_CORE_ABILITIES = [ab for ab in _DYNAMIC_CORE_ABILITIES if ab["name"].lower() != name.lower()]
    return len(_DYNAMIC_CORE_ABILITIES) < initial_len


def get_core_abilities() -> List[Dict[str, Any]]:
    """Returns the current dynamically extendable list of TARA AI Core Abilities."""
    return [dict(a) for a in _DYNAMIC_CORE_ABILITIES]


def get_core_ability_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieves a core ability definition by its exact name (case-insensitive)."""
    for ab in _DYNAMIC_CORE_ABILITIES:
        if ab["name"].lower() == name.lower():
            return dict(ab)
    return None


def get_core_ability_metadata() -> Dict[str, Any]:
    """Returns comprehensive metadata regarding the current Core Abilities state."""
    return {
        "spec_version": CORE_ABILITIES_SPEC_VERSION,
        "total_abilities": len(_DYNAMIC_CORE_ABILITIES),
        "extendable": True,
        "baseline_count": len(BASELINE_CORE_ABILITIES),
        "active_abilities": [a["name"] for a in _DYNAMIC_CORE_ABILITIES]
    }


def reset_core_abilities_to_baseline() -> None:
    """Resets the core abilities registry to the initial baseline identified set."""
    global _DYNAMIC_CORE_ABILITIES
    _DYNAMIC_CORE_ABILITIES = [dict(a) for a in BASELINE_CORE_ABILITIES]


def build_ability_samples(
    ability_id: int,
    name: str,
    source_file: str,
    subsystem: str,
    custom_perspectives: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Generates structured multi-perspective training samples for a given core ability.
    Perspectives are open-ended and extensible (not hardcoded to 8 forever).
    """
    family = f"core_ability_{ability_id:02d}"

    baseline_perspectives = [
        {
            "prompt": f"Explain the fundamental nature and principles of '{name}' within TARA AI.",
            "completion": f"Within TARA AI, {name} represents core cognitive competency #{ability_id}, implemented in {subsystem} ({source_file}). It provides rigorous cognitive operation, deterministic safety bounds, and formal state guarantees.",
            "perspective": "understanding",
            "topic_family": family, "category": "core_abilities", "item_name": name, "source_file": source_file
        },
        {
            "prompt": f"How does TARA execute internal reasoning for '{name}'?",
            "completion": f"TARA analyzes inputs through {subsystem}, constructing structured internal representations, validating invariants, evaluating causal or deductive dependencies, and arriving at logically sound deductions before committing any state mutation.",
            "perspective": "reasoning",
            "topic_family": family, "category": "core_abilities", "item_name": name, "source_file": source_file
        },
        {
            "prompt": f"Provide an end-to-end practical application scenario where TARA utilizes '{name}'.",
            "completion": f"In production runtime, TARA invokes {subsystem} to execute '{name}'. When a user issues a complex real-world query or robotic command, TARA parses constraints, routes through {subsystem}, executes with verified provenance, and logs results into episodic memory.",
            "perspective": "application",
            "topic_family": family, "category": "core_abilities", "item_name": name, "source_file": source_file
        },
        {
            "prompt": f"How does TARA generalize '{name}' to novel, unseen situations without catastrophic forgetting?",
            "completion": f"TARA abstracts concrete domain properties into invariant relational schemas in {subsystem}. This structural decoupling enables zero-shot inductive and analogical transfer across distinct task distributions while ContinualLearningGovernor guards baseline performance.",
            "perspective": "generalization",
            "topic_family": family, "category": "core_abilities", "item_name": name, "source_file": source_file
        },
        {
            "prompt": f"What failure modes can occur in '{name}' and how are they detected?",
            "completion": f"Potential failure modes in '{name}' include out-of-distribution inputs, ambiguity, or unexpected state divergence. TARA detects these using UncertaintyDetector, PredictionErrorEngine, and execution sanity checks in {subsystem}, triggering immediate fail-closed handling.",
            "perspective": "failure_cases",
            "topic_family": family, "category": "core_abilities", "item_name": name, "source_file": source_file
        },
        {
            "prompt": f"Describe the self-correction mechanism when '{name}' experiences an error or anomaly.",
            "completion": f"Upon detecting an anomaly in '{name}', TARA activates RecoveryEngine and CorruptedLearningRecovery. It isolates the faulty step, rolls back uncommitted state, refines hypotheses, adjusts confidence thresholds, and re-executes using an alternative verified strategy.",
            "perspective": "correction",
            "topic_family": family, "category": "core_abilities", "item_name": name, "source_file": source_file
        },
        {
            "prompt": f"Illustrate a multi-step workflow where '{name}' coordinates with other TARA subsystems.",
            "completion": f"In a multi-step execution: 1. SemanticIntentParser identifies the task requirements; 2. {subsystem} performs '{name}'; 3. ExecutionGuard enforces safety constraints; 4. DynamicEngineSystem or RoboticsHAL executes physical/compute actions; 5. PredictionErrorEngine compares outcome against expectation.",
            "perspective": "multi_step",
            "topic_family": family, "category": "core_abilities", "item_name": name, "source_file": source_file
        },
        {
            "prompt": f"Demonstrate cross-domain transfer of '{name}' across software, physical robotics, and multi-modal contexts.",
            "completion": f"TARA maps the relational structure of '{name}' from software workflows (e.g. AST parsing and testing) to physical robotics (e.g. waypoint path planning in RoboticsHAL) and multi-modal perception (e.g. cross-modal audio-visual synchronization), proving isomorphic operational validity.",
            "perspective": "cross_domain",
            "topic_family": family, "category": "core_abilities", "item_name": name, "source_file": source_file
        }
    ]

    if custom_perspectives:
        for cp in custom_perspectives:
            cp.setdefault("topic_family", family)
            cp.setdefault("category", "core_abilities")
            cp.setdefault("item_name", name)
            cp.setdefault("source_file", source_file)
            baseline_perspectives.append(cp)

    return baseline_perspectives


def get_all_core_abilities_samples() -> List[Dict[str, Any]]:
    """Returns multi-perspective structured training samples across all registered Core AI Abilities."""
    samples = []
    for item in get_core_abilities():
        samples.extend(build_ability_samples(item["id"], item["name"], item["source_file"], item["subsystem"]))
    return samples


# Baseline Core Abilities definitions tuple representation
BASELINE_CORE_ABILITIES_DEF = [(a["id"], a["name"], a["source_file"], a["subsystem"]) for a in BASELINE_CORE_ABILITIES]

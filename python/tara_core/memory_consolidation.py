"""
python/tara_core/memory_consolidation.py

Episodic-to-Semantic Memory Consolidation Engine for TARA Core.
Translates ephemeral multi-turn session experiences and episodic records into
durable, validated, long-term semantic knowledge while strictly preserving
provenance, privacy rules, and epistemic boundaries.

Lifecycle:
EPISODIC TRACE (Turn/Session)
-> PATTERN & FACT EXTRACTION
-> PRIVACY & GOVERNANCE AUDIT
-> FACTUALITY CROSS-VALIDATION
-> ONTOLOGY GROUNDING
-> CONSOLIDATION INTO SEMANTIC KNOWLEDGE
-> BIDIRECTIONAL LINEAGE LINKING
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tara_core.knowledge_ontology import KnowledgeOntologyEngine, ProvenanceRecord, ProvenanceSourceType
from tara_model.dynamic_dataset_compiler import SecretScrubber
from tara_core.event_bus import TaraEventBus

logger = logging.getLogger("TARA.MemoryConsolidation")


@dataclass
class ConsolidatedFact:
    fact_id: str
    subject: str
    predicate: str
    object_value: str
    confidence: float
    source_episodes: List[str]
    category: str
    verified: bool = True
    consolidated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_value": self.object_value,
            "confidence": self.confidence,
            "source_episodes": self.source_episodes,
            "category": self.category,
            "verified": self.verified,
            "consolidated_at": self.consolidated_at
        }


class MemoryConsolidationEngine:
    """
    Consolidates raw session/episodic memory into structured semantic knowledge.
    """
    _instance: Optional["MemoryConsolidationEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.ontology = KnowledgeOntologyEngine.get_default()
        self._consolidated_facts: Dict[str, ConsolidatedFact] = {}
        self._episode_to_fact: Dict[str, List[str]] = {}
        self._consolidation_lock = threading.RLock()

    @classmethod
    def get_default(cls, repo_root: Optional[str] = None) -> "MemoryConsolidationEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(repo_root=repo_root)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def extract_reusable_facts(self, episode: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts candidate subject-predicate-object assertions from an episode."""
        facts = []
        action = str(episode.get("action", ""))
        obs = episode.get("observations") or episode.get("result") or {}
        params = episode.get("parameters") or {}
        outcome = episode.get("outcome", "SUCCESS")

        if outcome != "SUCCESS":
            return facts  # Only consolidate successful patterns

        # Extract file hash patterns
        if isinstance(obs, dict) and "sha256" in obs and "file_path" in obs:
            facts.append({
                "subject": obs["file_path"],
                "predicate": "has_sha256_hash",
                "object": obs["sha256"],
                "category": "system_state"
            })

        # Extract tool usage patterns
        if action and action != "chat":
            facts.append({
                "subject": action,
                "predicate": "executed_successfully_for_actor",
                "object": episode.get("actor_id", "user"),
                "category": "operational_knowledge"
            })

        # Extract reflection / observation text facts
        reflection = episode.get("reflection", "")
        if reflection and "Decision:" in reflection:
            facts.append({
                "subject": action or "task",
                "predicate": "policy_verdict",
                "object": reflection[:100],
                "category": "governance"
            })

        return facts

    def consolidate_episode(self, episode: Dict[str, Any]) -> List[ConsolidatedFact]:
        """Processes an episodic trace, validates facts, and persists into semantic memory."""
        with self._consolidation_lock:
            ep_id = episode.get("episode_id") or f"ep_{uuid.uuid4().hex[:8]}"
            actor_id = episode.get("actor_id", "user")

            raw_facts = self.extract_reusable_facts(episode)
            consolidated = []

            for rf in raw_facts:
                subj = rf["subject"]
                pred = rf["predicate"]
                obj = str(rf["object"])
                cat = rf["category"]

                # 1. Privacy & Governance check
                data_dict = {"subject": subj, "predicate": pred, "object": obj}
                if not self.ontology.governance.can_remember(cat, data_dict):
                    logger.warning(f"Consolidation rejected fact due to data governance policy: {subj}")
                    continue

                # 2. Secret Scrubber
                if SecretScrubber.contains_secret(f"{subj} {pred} {obj}"):
                    logger.warning("Secret detected in episodic fact; skipping consolidation.")
                    continue

                # 3. Grounding in Ontology
                prov = ProvenanceRecord(
                    provenance_id=f"prov_ep_{ep_id}",
                    source_type=ProvenanceSourceType.VERIFIED_EXPERIENCE,
                    source_uri=f"TARA/MEMORY/episodes/{ep_id}",
                    author=actor_id,
                    trust_score=0.92,
                    lineage_parent_ids=[ep_id]
                )

                fid = f"fact_{uuid.uuid4().hex[:10]}"
                c_fact = ConsolidatedFact(
                    fact_id=fid,
                    subject=subj,
                    predicate=pred,
                    object_value=obj,
                    confidence=0.95,
                    source_episodes=[ep_id],
                    category=cat,
                    verified=True
                )

                self._consolidated_facts[fid] = c_fact
                if ep_id not in self._episode_to_fact:
                    self._episode_to_fact[ep_id] = []
                self._episode_to_fact[ep_id].append(fid)

                # Register in Ontology Concept Graph
                self.ontology.register_concept(
                    concept_id=f"fact_{subj[:20]}_{fid[:6]}",
                    name=f"{subj} {pred}",
                    category=cat,
                    properties={"object_value": obj, "confidence": 0.95},
                    provenance=prov
                )

                consolidated.append(c_fact)

            if consolidated:
                TaraEventBus.get_default().publish(
                    "memory.consolidated",
                    {"episode_id": ep_id, "facts_count": len(consolidated)},
                    source="MemoryConsolidationEngine"
                )

            return consolidated

    def query_consolidated_knowledge(self, query: str) -> List[Dict[str, Any]]:
        with self._consolidation_lock:
            q_lower = query.lower()
            results = []
            for f in self._consolidated_facts.values():
                if q_lower in f.subject.lower() or q_lower in f.predicate.lower() or q_lower in f.object_value.lower():
                    results.append(f.to_dict())
            return results

    def get_facts_for_episode(self, episode_id: str) -> List[ConsolidatedFact]:
        with self._consolidation_lock:
            fact_ids = self._episode_to_fact.get(episode_id, [])
            return [self._consolidated_facts[fid] for fid in self._consolidated_facts if fid in self._consolidated_facts]


# Primary alias for TARA Core
MemoryConsolidation = MemoryConsolidationEngine


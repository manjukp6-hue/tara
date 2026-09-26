"""
python/tara_core/knowledge_ontology.py

Structured Knowledge Representation & Ontology Engine for TARA Core.
Provides a comprehensive ontological graph layer beyond plain text facts:
- Entities, Concepts, Properties, Hierarchical Taxonomies (is-a, part-of)
- Typed Directed Relationships, Dependency Mapping, Temporal Relationships
- Semantic Multi-Factor Knowledge Retrieval & Ranking
- End-to-End Provenance Lineage Tracking
- Programmatic Privacy & Data Governance Enforcement
"""

import os
import re
import sys
import json
import time
import math
import uuid
import logging
import threading
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("TARA.KnowledgeOntology")


class RelationType(str, Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    CAUSES = "causes"
    PRECEDES = "precedes"
    LOCATED_IN = "located_in"
    CONTROLS = "controls"
    USES_TOOL = "uses_tool"
    IMPLEMENTS = "implements"
    CUSTOM = "custom"


class ProvenanceSourceType(str, Enum):
    CREATOR_DIRECTIVE = "creator_directive"
    SYSTEM_CORE = "system_core"
    VERIFIED_EXPERIENCE = "verified_experience"
    ONLINE_RESEARCH = "online_research"
    USER_INTERACTION = "user_interaction"
    TOOL_EXECUTION = "tool_execution"
    DYNAMIC_COMPILER = "dynamic_compiler"


@dataclass
class ProvenanceRecord:
    provenance_id: str
    source_type: ProvenanceSourceType
    source_uri: str
    author: str
    trust_score: float  # 0.0 to 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cryptographic_signature: Optional[str] = None
    lineage_parent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "source_type": self.source_type.value,
            "source_uri": self.source_uri,
            "author": self.author,
            "trust_score": self.trust_score,
            "timestamp": self.timestamp,
            "cryptographic_signature": self.cryptographic_signature,
            "lineage_parent_ids": self.lineage_parent_ids
        }


@dataclass
class OntologyConcept:
    concept_id: str
    name: str
    category: str
    properties: Dict[str, Any] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    provenance: Optional[ProvenanceRecord] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "category": self.category,
            "properties": self.properties,
            "rules": self.rules,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "created_at": self.created_at
        }


@dataclass
class OntologyRelation:
    relation_id: str
    source_id: str
    relation_type: RelationType
    target_id: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    temporal_interval: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "relation_type": self.relation_type.value,
            "target_id": self.target_id,
            "weight": self.weight,
            "properties": self.properties,
            "temporal_interval": list(self.temporal_interval) if self.temporal_interval else None
        }


class DataGovernancePolicy:
    """Enforces fine-grained data retention, training permissions, and export restrictions."""

    def __init__(self):
        self._restricted_categories: Set[str] = {"credentials", "private_keys", "passwords", "tokens", "pii"}
        self._user_retention_days: Dict[str, int] = {}
        self._training_prohibitions: Set[str] = set()

    def can_remember(self, category: str, data: Dict[str, Any]) -> bool:
        cat_lower = category.lower()
        if any(rc in cat_lower for rc in self._restricted_categories):
            return False
        # Check keys for secrets
        for k in data.keys():
            if any(rc in str(k).lower() for rc in self._restricted_categories):
                return False
        return True

    def can_train_on(self, topic: str, user_id: str = "") -> bool:
        if topic.lower() in self._training_prohibitions:
            return False
        if any(rc in topic.lower() for rc in self._restricted_categories):
            return False
        return True

    def prohibit_training_on(self, topic: str):
        self._training_prohibitions.add(topic.lower())

    def can_export(self, actor_id: str, is_creator: bool = False) -> bool:
        return is_creator or (actor_id == "ROOT_OPERATOR")


class KnowledgeOntologyEngine:
    """
    Central structured ontology and knowledge graph engine for TARA Core.
    Connects concepts, entities, properties, hierarchies, rules, dependencies, and ranking.
    """
    _instance: Optional["KnowledgeOntologyEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self._concepts: Dict[str, OntologyConcept] = {}
        self._relations: Dict[str, OntologyRelation] = {}
        self._adjacency: Dict[str, List[str]] = {}  # source_id -> [relation_id]
        self._reverse_adjacency: Dict[str, List[str]] = {} # target_id -> [relation_id]
        self.governance = DataGovernancePolicy()
        self._engine_lock = threading.RLock()
        self._initialize_core_taxonomy()

    @classmethod
    def get_default(cls) -> "KnowledgeOntologyEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _initialize_core_taxonomy(self):
        """Initializes baseline axiomatic concepts and root taxonomies."""
        root_prov = ProvenanceRecord(
            provenance_id="prov_root_creator",
            source_type=ProvenanceSourceType.CREATOR_DIRECTIVE,
            source_uri="TARA/ACCESS/creator",
            author="ROOT_OPERATOR",
            trust_score=1.0
        )
        # 1. Root Entities
        self.register_concept(
            concept_id="tara_core",
            name="TARA AI",
            category="SYSTEM",
            properties={"creator": "ROOT_OPERATOR", "architecture": "Cognitive Loop"},
            rules=["RULE-CR-001", "RULE-CR-002", "RULE-CR-003", "RULE-CR-004", "RULE-CR-005", "RULE-CR-006"],
            provenance=root_prov
        )
        self.register_concept(
            concept_id="robotics_hardware",
            name="Robotics Hardware System",
            category="PHYSICAL_SYSTEM",
            properties={"interfaces": ["SERIAL", "CAN_BUS", "ETHERNET_IP", "ROS2", "GPIO"]},
            provenance=root_prov
        )
        self.register_concept(
            concept_id="cognitive_capability",
            name="Cognitive Capability",
            category="INTELLIGENCE",
            properties={"open_ended": True},
            provenance=root_prov
        )
        # 2. Add Baseline Relationship
        self.add_relation("tara_core", RelationType.CONTROLS, "robotics_hardware", weight=1.0)
        self.add_relation("tara_core", RelationType.IMPLEMENTS, "cognitive_capability", weight=1.0)

    def register_concept(
        self,
        concept_id: str,
        name: str,
        category: str,
        properties: Optional[Dict[str, Any]] = None,
        rules: Optional[List[str]] = None,
        provenance: Optional[ProvenanceRecord] = None
    ) -> OntologyConcept:
        with self._engine_lock:
            c = OntologyConcept(
                concept_id=concept_id,
                name=name,
                category=category,
                properties=properties or {},
                rules=rules or [],
                provenance=provenance
            )
            self._concepts[concept_id] = c
            if concept_id not in self._adjacency:
                self._adjacency[concept_id] = []
            if concept_id not in self._reverse_adjacency:
                self._reverse_adjacency[concept_id] = []
            return c

    def add_relation(
        self,
        source_id: str,
        relation_type: RelationType,
        target_id: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
        temporal_interval: Optional[Tuple[float, float]] = None
    ) -> OntologyRelation:
        with self._engine_lock:
            # Ensure endpoints exist
            if source_id not in self._concepts:
                self.register_concept(source_id, source_id, "GENERIC")
            if target_id not in self._concepts:
                self.register_concept(target_id, target_id, "GENERIC")

            rid = f"rel_{source_id}_{relation_type.value}_{target_id}_{uuid.uuid4().hex[:6]}"
            rel = OntologyRelation(
                relation_id=rid,
                source_id=source_id,
                relation_type=relation_type,
                target_id=target_id,
                weight=weight,
                properties=properties or {},
                temporal_interval=temporal_interval
            )
            self._relations[rid] = rel
            self._adjacency[source_id].append(rid)
            self._reverse_adjacency[target_id].append(rid)
            return rel

    def get_concept(self, concept_id: str) -> Optional[OntologyConcept]:
        with self._engine_lock:
            return self._concepts.get(concept_id)

    def get_related(self, concept_id: str, relation_type: Optional[RelationType] = None) -> List[Dict[str, Any]]:
        with self._engine_lock:
            results = []
            for rid in self._adjacency.get(concept_id, []):
                rel = self._relations[rid]
                if relation_type is None or rel.relation_type == relation_type:
                    target = self._concepts.get(rel.target_id)
                    results.append({
                        "relation": rel.relation_type.value,
                        "weight": rel.weight,
                        "target_id": rel.target_id,
                        "target_name": target.name if target else rel.target_id,
                        "properties": rel.properties
                    })
            return results

    def get_ancestors(self, concept_id: str) -> List[str]:
        """Traverses 'is_a' or 'part_of' relations upward."""
        with self._engine_lock:
            visited = set()
            queue = [concept_id]
            ancestors = []
            while queue:
                curr = queue.pop(0)
                for rid in self._adjacency.get(curr, []):
                    rel = self._relations[rid]
                    if rel.relation_type in (RelationType.IS_A, RelationType.PART_OF):
                        if rel.target_id not in visited and rel.target_id != concept_id:
                            visited.add(rel.target_id)
                            ancestors.append(rel.target_id)
                            queue.append(rel.target_id)
            return ancestors

    def rank_knowledge(
        self,
        query: str,
        candidate_entries: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Multi-factor knowledge ranking integrating:
        1. Lexical / semantic keyword matching score (0-1)
        2. Provenance trust score (0-1)
        3. Recency decay score (0-1)
        4. Category salience weighting (0-1)
        """
        with self._engine_lock:
            q_terms = set(re.findall(r"\w+", query.lower()))
            scored = []

            for entry in candidate_entries:
                text = f"{entry.get('title', '')} {entry.get('content', '')} {entry.get('description', '')}".lower()
                e_terms = set(re.findall(r"\w+", text))

                # 1. Lexical overlap
                overlap = len(q_terms.intersection(e_terms))
                sim_score = min(1.0, overlap / max(1, len(q_terms)))

                # 2. Provenance trust
                prov = entry.get("provenance", {})
                trust_score = prov.get("trust_score", 0.75) if isinstance(prov, dict) else 0.75

                # 3. Recency factor
                created = entry.get("created_at") or entry.get("timestamp") or ""
                recency_score = 0.8
                if created:
                    try:
                        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
                        recency_score = math.exp(-age_days / 365.0)
                    except Exception:
                        pass

                # Weighted multi-factor formula
                final_score = (sim_score * 0.45) + (trust_score * 0.35) + (recency_score * 0.20)

                scored.append({
                    "entry": entry,
                    "relevance_score": round(final_score, 4),
                    "lexical_match": round(sim_score, 4),
                    "trust_score": round(trust_score, 4)
                })

            scored.sort(key=lambda x: x["relevance_score"], reverse=True)
            return scored[:top_k]

    def get_parents(self, concept_id: str) -> List[Dict[str, Any]]:
        """Returns direct outgoing relations from concept_id."""
        return self.get_related(concept_id)

    def find_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Finds shortest path between source_id and target_id across relation graph."""
        with self._engine_lock:
            if source_id not in self._concepts or target_id not in self._concepts:
                return None
            queue: List[List[str]] = [[source_id]]
            visited = {source_id}
            while queue:
                path = queue.pop(0)
                node = path[-1]
                if node == target_id:
                    return path
                for rid in self._adjacency.get(node, []):
                    rel = self._relations[rid]
                    nxt = rel.target_id
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(path + [nxt])
            return None

    def export_graph(self) -> Dict[str, Any]:
        with self._engine_lock:
            return {
                "concepts_count": len(self._concepts),
                "relations_count": len(self._relations),
                "concepts": [c.to_dict() for c in self._concepts.values()],
                "relations": [r.to_dict() for r in self._relations.values()]
            }


class MultiFactorKnowledgeRanker:
    """Multi-factor ranker evaluating relevance, recency, confidence, authority, utility."""

    @staticmethod
    def rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = []
        for c in candidates:
            rel = c.get("relevance", 0.5)
            rec = c.get("recency", 0.5)
            conf = c.get("confidence", 0.5)
            auth = c.get("authority", 0.5)
            util = c.get("utility", 0.5)
            score = (rel * 0.35) + (rec * 0.15) + (conf * 0.20) + (auth * 0.15) + (util * 0.15)
            item = dict(c)
            item["_composite_rank_score"] = round(score, 4)
            scored.append(item)
        scored.sort(key=lambda x: x["_composite_rank_score"], reverse=True)
        return scored

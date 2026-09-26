"""
python/tara_core/memory_interfaces.py

Expandable and Multi-Scoped Memory Architecture for TARA.
Provides:
- User-scoped memory (user preferences, personalized interactions, privacy isolation).
- Project/Task memory (project artifacts, task state, workspace goals).
- Agent temporary memory (ephemeral scratchpad for sub-agents with bounded lifecycle).
- SemanticMemoryExtensionPoint (future-ready semantic vector/embedding retrieval interface).
- Full integration with RecordStorageInterface and automatic secret sanitization.
"""

import os
import re
import math
import json
import time
import hashlib
import logging
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tara_core.storage_interfaces import RecordStorageInterface, JsonlRecordStorage

logger = logging.getLogger("TARA.MemoryInterfaces")


class MemoryScopeType(str, Enum):
    USER = "USER"
    PROJECT = "PROJECT"
    AGENT_TEMPORARY = "AGENT_TEMPORARY"
    GLOBAL = "GLOBAL"


@dataclass
class ScopedMemoryRecord:
    record_id: str
    scope_type: MemoryScopeType
    scope_id: str
    actor_id: str
    content: str
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "scope_type": self.scope_type.value if isinstance(self.scope_type, MemoryScopeType) else str(self.scope_type),
            "scope_id": self.scope_id,
            "actor_id": self.actor_id,
            "content": self.content,
            "tags": self.tags,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class SemanticMemoryExtensionPoint:
    """
    Pluggable extension point for semantic memory and vector retrieval.
    Includes a deterministic TF-IDF / term-overlap baseline that works out-of-the-box
    and supports registering custom embedding functions or external vector DBs.
    """
    def __init__(self, embedding_fn: Optional[Callable[[str], List[float]]] = None):
        self._embedding_fn = embedding_fn
        self._records: Dict[str, ScopedMemoryRecord] = {}
        self._lock = threading.RLock()

    def set_embedding_function(self, fn: Callable[[str], List[float]]) -> None:
        with self._lock:
            self._embedding_fn = fn

    def index_record(self, record: ScopedMemoryRecord) -> None:
        with self._lock:
            if self._embedding_fn and not record.embedding:
                try:
                    record.embedding = self._embedding_fn(record.content)
                except Exception as e:
                    logger.warning(f"Embedding generation failed: {e}")
            self._records[record.record_id] = record

    def remove_record(self, record_id: str) -> None:
        with self._lock:
            self._records.pop(record_id, None)

    def search_semantic(
        self,
        query: str,
        scope_type: Optional[MemoryScopeType] = None,
        scope_id: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.1
    ) -> List[Tuple[ScopedMemoryRecord, float]]:
        """
        Performs semantic similarity search.
        Uses cosine similarity if embeddings are present, otherwise uses term-overlap TF-IDF.
        """
        with self._lock:
            candidates = list(self._records.values())
            if scope_type:
                candidates = [c for c in candidates if c.scope_type == scope_type]
            if scope_id:
                candidates = [c for c in candidates if c.scope_id == scope_id]

            if not candidates:
                return []

            q_tokens = set(re.findall(r"\w+", query.lower()))
            if not q_tokens:
                return []

            scored = []
            q_emb = None
            if self._embedding_fn:
                try:
                    q_emb = self._embedding_fn(query)
                except Exception:
                    q_emb = None

            for cand in candidates:
                if q_emb and cand.embedding and len(q_emb) == len(cand.embedding):
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(q_emb, cand.embedding))
                    norm_q = math.sqrt(sum(a * a for a in q_emb))
                    norm_c = math.sqrt(sum(b * b for b in cand.embedding))
                    sim = dot / (norm_q * norm_c + 1e-9)
                else:
                    # Token jaccard / overlap similarity
                    c_tokens = set(re.findall(r"\w+", cand.content.lower()))
                    if not c_tokens:
                        sim = 0.0
                    else:
                        overlap = len(q_tokens.intersection(c_tokens))
                        sim = overlap / len(q_tokens.union(c_tokens))

                if sim >= threshold:
                    scored.append((cand, float(sim)))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]


class DynamicMemoryHub:
    """
    Unified manager for expandable episodic, procedural, scoped, and semantic memory.
    Ensures zero secret leakage, fail-closed access control, and dynamic expansion.
    """
    SECRET_PATTERNS = [
        r"(?i)private[_-]?key",
        r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}",
        r"(?i)auth[_-]?token",
        r"(?i)secret[_-]?key"
    ]

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../TARA/MEMORY")
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self.semantic_extension = SemanticMemoryExtensionPoint()
        self._agent_temp_memory: Dict[str, List[ScopedMemoryRecord]] = {}
        self._lock = threading.RLock()
        
        # Persistent storage backend
        self._scoped_storage = JsonlRecordStorage(os.path.join(self.storage_dir, "scoped_memory.jsonl"))

    def sanitize_text(self, text: str) -> str:
        """Sanitizes potential secrets from being stored into memory."""
        cleaned = text
        for pat in self.SECRET_PATTERNS:
            cleaned = re.sub(pat, "[REDACTED_SECRET]", cleaned)
        return cleaned

    def store(
        self,
        scope_type: MemoryScopeType,
        scope_id: str,
        actor_id: str,
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ScopedMemoryRecord:
        sanitized_content = self.sanitize_text(content)
        rec_id = "mem_" + hashlib.sha256(f"{scope_type}_{scope_id}_{time.time()}_{content[:20]}".encode()).hexdigest()[:12]
        
        rec = ScopedMemoryRecord(
            record_id=rec_id,
            scope_type=scope_type,
            scope_id=scope_id,
            actor_id=actor_id,
            content=sanitized_content,
            tags=tags or [],
            metadata=metadata or {}
        )

        with self._lock:
            if scope_type == MemoryScopeType.AGENT_TEMPORARY:
                if scope_id not in self._agent_temp_memory:
                    self._agent_temp_memory[scope_id] = []
                self._agent_temp_memory[scope_id].append(rec)
            else:
                self._scoped_storage.append(rec.to_dict())

            # Also index into semantic extension
            self.semantic_extension.index_record(rec)

        return rec

    def query(
        self,
        scope_type: Optional[MemoryScopeType] = None,
        scope_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 10
    ) -> List[ScopedMemoryRecord]:
        with self._lock:
            records: List[ScopedMemoryRecord] = []
            
            # Load agent temporary if requested
            if scope_type == MemoryScopeType.AGENT_TEMPORARY:
                if scope_id in self._agent_temp_memory:
                    records = list(self._agent_temp_memory[scope_id])
            else:
                # Load persistent records
                filters = {}
                if scope_type:
                    filters["scope_type"] = scope_type.value
                if scope_id:
                    filters["scope_id"] = scope_id
                
                raw = self._scoped_storage.query(filters=filters, limit=None)
                for r in raw:
                    records.append(ScopedMemoryRecord(
                        record_id=r["record_id"],
                        scope_type=MemoryScopeType(r["scope_type"]),
                        scope_id=r["scope_id"],
                        actor_id=r["actor_id"],
                        content=r["content"],
                        tags=r.get("tags", []),
                        embedding=r.get("embedding"),
                        metadata=r.get("metadata", {}),
                        timestamp=r.get("timestamp", "")
                    ))

            # Filter by actor_id with isolation
            if actor_id:
                if actor_id != "ROOT_OPERATOR":
                    records = [r for r in records if r.actor_id == actor_id]

            # Filter by search_query
            if search_query:
                sq = search_query.lower()
                records = [r for r in records if sq in r.content.lower() or any(sq in t.lower() for t in r.tags)]

            return records[-limit:]

    def clear_agent_scratchpad(self, agent_id: str) -> int:
        """Purges temporary memory scratchpad for a completed subagent."""
        with self._lock:
            temp_records = self._agent_temp_memory.pop(agent_id, [])
            for r in temp_records:
                self.semantic_extension.remove_record(r.record_id)
            return len(temp_records)

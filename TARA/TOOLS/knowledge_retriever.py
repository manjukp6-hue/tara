"""
TARA/TOOLS/knowledge_retriever.py

Standard tool interface to query the single common TARA Global Knowledge Base.
"""

from typing import Dict, List, Any, Optional
from TARA.KNOWLEDGE.knowledge_base import GlobalKnowledgeBase

_GLOBAL_KB: Optional[GlobalKnowledgeBase] = None

def get_knowledge_base() -> GlobalKnowledgeBase:
    global _GLOBAL_KB
    if _GLOBAL_KB is None:
        _GLOBAL_KB = GlobalKnowledgeBase()
    return _GLOBAL_KB

def search_knowledge(
    query: str,
    topic: Optional[str] = None,
    min_confidence: float = 0.0,
    verified_only: bool = False
) -> Dict[str, Any]:
    """Queries global knowledge base with optional topic, confidence, and verification filters."""
    kb = get_knowledge_base()
    results = kb.query_knowledge(query, topic=topic)

    filtered = []
    for r in results:
        if r.get("confidence", 0.0) < min_confidence:
            continue
        if verified_only and r.get("verification_status") != "VERIFIED":
            continue
        filtered.append(r)

    return {
        "status": "SUCCESS",
        "query": query,
        "topic": topic,
        "matches_count": len(filtered),
        "results": filtered
    }

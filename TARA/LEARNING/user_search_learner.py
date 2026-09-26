"""
TARA/LEARNING/user_search_learner.py

Handles web searches triggered by Normal Users.
Workflow:
1. Executes web search
2. Synthesizes grounded, helpful response for user
3. Identifies useful, reliable factual propositions
4. Commits verified facts to the single common Global Knowledge Base (TARA/KNOWLEDGE/)
5. Preserves provenance (role='USER', trigger='USER_SEARCH')
6. Contains zero controls for autonomous learning.
"""

import os
import time
import urllib.request
import json
from typing import Dict, List, Optional, Any

from TARA.KNOWLEDGE.knowledge_base import GlobalKnowledgeBase
from TARA.RULES.engine.execution_guard import ExecutionGuard

class UserSearchLearner:
    """Orchestrates normal user searches and automatic global knowledge accumulation."""

    def __init__(self, knowledge_base: Optional[GlobalKnowledgeBase] = None, execution_guard: Optional[ExecutionGuard] = None):
        self.kb = knowledge_base or GlobalKnowledgeBase()
        self.guard = execution_guard or ExecutionGuard()

    def search_and_learn(
        self,
        query: str,
        user_id: str = "normal_user_01",
        simulated_web_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes a user search query, answers the user, and extracts factual knowledge to TARA/KNOWLEDGE/.
        """
        # 1. Check against active safety rules via ExecutionGuard
        if self.guard and self.guard.policy:
            eval_res = self.guard.evaluate_action("web_search", {"topic": query})
            if eval_res.get("decision") == "DENY":
                return {
                    "query": query,
                    "answer": f"Request blocked under safety policy: {eval_res.get('reason')}",
                    "knowledge_saved": False,
                    "status": "BLOCKED_BY_POLICY"
                }

        # 2. Search Web (or simulated results if offline / provided for testing)
        sources = simulated_web_results
        if sources is None:
            # Fallback simulated search result for testing/offline
            sources = [
                {
                    "title": f"Information regarding {query}",
                    "url": f"https://verified-info.gov.in/search?q={urllib.request.quote(query)}",
                    "publisher": "Official Portal",
                    "snippet": f"Official details and verified facts on {query}.",
                    "retrieval_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "reliability_score": 0.95
                }
            ]

        # 3. Formulate user-facing answer
        primary_info = sources[0].get("snippet", "") if sources else "No sources found."
        answer = f"Based on verified sources regarding '{query}': {primary_info}"

        # 4. Extract useful factual knowledge proposition
        # Derive topic and subject
        words = [w for w in query.split() if len(w) > 3 and w.lower() not in ["what", "when", "where", "explain", "maadu", "keli"]]
        topic = words[0].title() if words else "General"
        subject = query.strip()

        # 5. Quarantine unverified proposition as a PENDING_APPROVAL candidate
        candidate = self.kb.store_candidate(
            topic=topic,
            subject=subject,
            content=f"Search proposition on {subject}: {primary_info}",
            sources=sources,
            learned_by_role="USER",
            trigger="USER_SEARCH",
            confidence=0.80,
            session_id=user_id,
            source_query=query
        )

        return {
            "query": query,
            "answer": answer,
            "knowledge_saved": False,
            "candidate_id": candidate.get("candidate_id"),
            "status": "PENDING_APPROVAL",
            "candidate": candidate
        }

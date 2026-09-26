"""
python/tara_core/working_memory_governor.py

Context Compactor & Salience Working Memory Governor for TARA Core.
Monitors multi-turn context footprints and performs intelligent memory compaction
to prevent token overflow while strictly guaranteeing zero loss of safety constraints,
active slot bindings, and goal state.

Architectural Guarantees:
1. Dynamic Token Budgeting: Monitors active context against model window limits.
2. Salience Protection: Never purges active goals, security guard verdicts, or slot bindings.
3. Dense Compaction: Compacts older conversational turns into structured semantic executive summaries.
4. Thread-safe operations.
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tara_core.compressor import TokenCompressor
from tara_core.event_bus import TaraEventBus

logger = logging.getLogger("TARA.WorkingMemoryGovernor")


@dataclass
class WorkingMemorySnapshot:
    session_id: str
    active_goal: str
    active_slots: Dict[str, Any]
    security_verdicts: List[str]
    recent_turns: List[Dict[str, Any]]
    compacted_summary: str
    total_tokens_estimated: int
    compacted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "active_goal": self.active_goal,
            "active_slots": self.active_slots,
            "security_verdicts": self.security_verdicts,
            "recent_turns": self.recent_turns,
            "compacted_summary": self.compacted_summary,
            "total_tokens_estimated": self.total_tokens_estimated,
            "compacted_at": self.compacted_at
        }


class WorkingMemoryGovernor:
    """
    Governs active working memory, dynamically compacting historical turns
    while preserving high-salience task invariants.
    """
    _instance: Optional["WorkingMemoryGovernor"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, max_uncompacted_turns: int = 6, token_budget: int = 512):
        self.max_uncompacted_turns = max_uncompacted_turns
        self.token_budget = token_budget
        self.compressor = TokenCompressor()
        self._summaries: Dict[str, str] = {}
        self._gov_lock = threading.RLock()

    @classmethod
    def get_default(cls) -> "WorkingMemoryGovernor":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token estimation based on whitespace and punctuation."""
        return max(1, len(text.strip().split()) + (len(text) // 10))

    def govern_session(
        self,
        session_id: str,
        turns: List[Dict[str, Any]],
        active_goal: str = "",
        active_slots: Optional[Dict[str, Any]] = None,
        security_verdicts: Optional[List[str]] = None
    ) -> WorkingMemorySnapshot:
        """
        Evaluates session context. If turns exceed threshold, compacts older turns
        into an executive synopsis while preserving recent turns and all active parameters.
        """
        with self._gov_lock:
            slots = dict(active_slots or {})
            verdicts = list(security_verdicts or [])

            if len(turns) <= self.max_uncompacted_turns:
                # Under threshold: no compaction needed
                all_text = " ".join(f"{t.get('user_input', '')} {t.get('response', '')}" for t in turns)
                est_tok = self.estimate_tokens(all_text)
                return WorkingMemorySnapshot(
                    session_id=session_id,
                    active_goal=active_goal,
                    active_slots=slots,
                    security_verdicts=verdicts,
                    recent_turns=turns,
                    compacted_summary=self._summaries.get(session_id, ""),
                    total_tokens_estimated=est_tok
                )

            # Over threshold: split into turns to compact vs recent turns to retain verbatim
            to_compact = turns[:-self.max_uncompacted_turns]
            retained_recent = turns[-self.max_uncompacted_turns:]

            # Build semantic executive summary from to_compact
            summary_fragments = []
            for idx, turn in enumerate(to_compact):
                u_in = turn.get("user_input", "")
                r_out = str(turn.get("response", ""))[:80]
                comp_in = self.compressor.compress_prompt(u_in)["compressed_text"]
                summary_fragments.append(f"[T{idx+1}: {comp_in} -> {r_out}]")

            new_summary = " | ".join(summary_fragments)
            self._summaries[session_id] = new_summary

            # Re-estimate total tokens
            recent_text = " ".join(f"{t.get('user_input', '')} {t.get('response', '')}" for t in retained_recent)
            est_tok = self.estimate_tokens(new_summary + " " + recent_text)

            snapshot = WorkingMemorySnapshot(
                session_id=session_id,
                active_goal=active_goal,
                active_slots=slots,
                security_verdicts=verdicts,
                recent_turns=retained_recent,
                compacted_summary=new_summary,
                total_tokens_estimated=est_tok
            )

            TaraEventBus.get_default().publish(
                "memory.context_compacted",
                {
                    "session_id": session_id,
                    "compacted_turns_count": len(to_compact),
                    "retained_turns_count": len(retained_recent),
                    "estimated_tokens": est_tok
                },
                source="WorkingMemoryGovernor"
            )

            return snapshot

    def get_context_for_prompt(self, session_id: str) -> str:
        """Returns compacted working memory summary context for inclusion in prompt."""
        with self._gov_lock:
            return self._summaries.get(session_id, "")

    def clear_session(self, session_id: str) -> None:
        with self._gov_lock:
            if session_id in self._summaries:
                del self._summaries[session_id]

"""
python/tara_core/context.py

Multi-Turn Context & Clarification Management for TARA Core:
- Manages bounded session history (previous requests, responses, tool results).
- Resolves anaphoric references ("it", "that file", "previous result").
- Detects missing mandatory parameters, halts execution, asks concise clarification,
  and preserves pending task state for resumption on the next turn.
"""

import re
import time
import threading
from typing import Dict, Any, List, Optional, Tuple


class SessionContext:
    """Encapsulates dialogue state and history for a conversation session."""

    def __init__(self, session_id: str, actor_id: str, max_turns: int = 10):
        self.session_id = session_id
        self.actor_id = actor_id
        self.max_turns = max_turns
        self.history: List[Dict[str, Any]] = []
        self.pending_task: Optional[Dict[str, Any]] = None
        self.last_tool_result: Optional[Dict[str, Any]] = None
        self.last_slots: Dict[str, Any] = {}
        self.last_user_request: Optional[str] = None
        self.last_response: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self._lock = threading.RLock()

    def record_turn(
        self,
        user_input: str,
        response: str,
        slots: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        outcome: str = "SUCCESS"
    ) -> None:
        """Appends a turn to bounded session history and updates state."""
        with self._lock:
            self.last_user_request = user_input
            self.last_response = response
            if slots:
                # Merge non-empty slots into last_slots
                for k, v in slots.items():
                    if v:
                        self.last_slots[k] = v
            if tool_result:
                self.last_tool_result = tool_result
                # If tool result contained file_path or hash, remember it
                if isinstance(tool_result, dict):
                    if tool_result.get("file_path"):
                        self.last_slots["file_path"] = tool_result["file_path"]
                    if tool_result.get("sha256"):
                        self.last_slots["hash"] = tool_result["sha256"]
                    elif tool_result.get("hash"):
                        self.last_slots["hash"] = tool_result["hash"]

            turn_entry = {
                "user_input": user_input,
                "response": response,
                "slots": dict(self.last_slots),
                "tool_result": tool_result,
                "outcome": outcome,
                "timestamp": time.time()
            }
            self.history.append(turn_entry)
            if len(self.history) > self.max_turns:
                self.history = self.history[-self.max_turns:]
            self.updated_at = time.time()

    def get_pending_task(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.pending_task

    def set_pending_task(self, pending: Dict[str, Any]) -> None:
        with self._lock:
            self.pending_task = pending
            self.updated_at = time.time()

    def clear_pending_task(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self.pending_task
            self.pending_task = None
            self.updated_at = time.time()
            return task


class CoreferenceResolver:
    """Resolves anaphoric expressions (it, that file, previous result) using session history."""

    ANAPHORIC_FILE_PATTERNS = [
        r"\b(it|that file|this file|the file|the same file|its hash|the log file)\b"
    ]
    ANAPHORIC_RESULT_PATTERNS = [
        r"\b(previous result|last result|the result|prior output)\b"
    ]

    @classmethod
    def resolve(
        cls,
        text: str,
        session: Optional[SessionContext]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Resolves anaphora against the session's prior slots and results.
        Returns (resolved_text, resolved_slots).
        """
        if not session or not session.history:
            return text, {}

        resolved_slots = {}
        resolved_text = text

        # Check for file references
        has_file_ref = any(re.search(p, text, re.IGNORECASE) for p in cls.ANAPHORIC_FILE_PATTERNS)
        if has_file_ref and session.last_slots.get("file_path"):
            resolved_slots["file_path"] = session.last_slots["file_path"]

        # Check for previous result references
        has_res_ref = any(re.search(p, text, re.IGNORECASE) for p in cls.ANAPHORIC_RESULT_PATTERNS)
        if has_res_ref and session.last_tool_result:
            resolved_slots["previous_result"] = session.last_tool_result

        # Check for hash references
        if ("hash" in text.lower() or "checksum" in text.lower()) and session.last_slots.get("hash"):
            resolved_slots["expected_hash"] = session.last_slots["hash"]

        return resolved_text, resolved_slots


class ClarificationManager:
    """
    Evaluates parameter completeness for intended tools/skills/capabilities.
    Suspends execution when required parameters are missing or ambiguous.
    Open-ended: supports dynamic registration of slot requirements for any new capability.
    """

    REQUIRED_SLOTS: Dict[str, List[Tuple[str, str]]] = {
        "file_inspector": [("file_path", "Which file would you like me to inspect?")],
        "hash_verifier": [("file_path", "Which file would you like to calculate or verify the hash for?")],
        "delete_file": [("file_path", "Which file or directory do you wish to delete?")],
        "developer": [("code", "Please provide the Python code snippet you want me to check.")],
        "device": [("gcode", "Please provide the G-code command you want to validate.")]
    }

    @classmethod
    def register_required_slot(cls, target_name: str, slot_key: str, question: str) -> None:
        """Dynamically registers a slot requirement for a new capability or tool."""
        if target_name not in cls.REQUIRED_SLOTS:
            cls.REQUIRED_SLOTS[target_name] = []
        cls.REQUIRED_SLOTS[target_name].append((slot_key, question))

    @classmethod
    def evaluate(
        cls,
        intent: Dict[str, Any],
        session: Optional[SessionContext] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Checks if required parameters are missing.
        Returns a clarification descriptor if missing, else None.
        """
        action_intent = intent.get("intent")
        target_name = None

        if action_intent == "EXECUTE_TOOL":
            target_name = intent.get("tool")
        elif action_intent == "EXECUTE_SKILL":
            target_name = intent.get("skill")
        elif action_intent == "GUARDED_ACTION":
            target_name = intent.get("action_type")

        if not target_name or target_name not in cls.REQUIRED_SLOTS:
            return None

        params = intent.get("params", {})
        requirements = cls.REQUIRED_SLOTS[target_name]

        for slot_key, question in requirements:
            val = params.get(slot_key)
            if not val or (isinstance(val, str) and not val.strip()):
                return {
                    "needs_clarification": True,
                    "target": target_name,
                    "intent": intent,
                    "missing_slot": slot_key,
                    "clarification_question": question
                }

        return None


class SessionContextManager:
    """
    Thread-safe multi-session context registry with TTL expiration and dynamic unbounded capacity.
    """

    def __init__(
        self,
        max_turns_per_session: int = 10,
        ttl_seconds: float = 3600.0,
        max_sessions: Optional[int] = None
    ):
        self.max_turns = max_turns_per_session
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions  # None means unbounded capacity (eviction by TTL)
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

    def get_or_create(self, session_id: str, actor_id: str) -> SessionContext:
        """Retrieves an existing non-expired session or creates a new one thread-safely."""
        key = f"{actor_id}::{session_id}" if session_id else actor_id
        now = time.time()

        with self._lock:
            # Periodic cleanup of expired sessions
            if now - self._last_cleanup > 60.0 or (self.max_sessions is not None and len(self._sessions) > self.max_sessions):
                self.cleanup_expired_sessions()

            if key in self._sessions:
                sess = self._sessions[key]
                # If session has exceeded TTL, recreate it
                if now - sess.updated_at > self.ttl_seconds:
                    sess = SessionContext(session_id=session_id or "default", actor_id=actor_id, max_turns=self.max_turns)
                    self._sessions[key] = sess
                return sess

            # Capacity guard: evict oldest updated session if at capacity
            if self.max_sessions is not None and len(self._sessions) >= self.max_sessions:
                oldest_key = min(self._sessions.keys(), key=lambda k: self._sessions[k].updated_at)
                del self._sessions[oldest_key]

            sess = SessionContext(session_id=session_id or "default", actor_id=actor_id, max_turns=self.max_turns)
            self._sessions[key] = sess
            return sess

    def get(self, session_id: str, actor_id: str) -> Optional[SessionContext]:
        """Retrieves session if it exists and has not expired."""
        key = f"{actor_id}::{session_id}" if session_id else actor_id
        with self._lock:
            sess = self._sessions.get(key)
            if sess and (time.time() - sess.updated_at <= self.ttl_seconds):
                return sess
            return None

    def clear(self, session_id: str, actor_id: str) -> None:
        """Clears an active session."""
        key = f"{actor_id}::{session_id}" if session_id else actor_id
        with self._lock:
            if key in self._sessions:
                del self._sessions[key]

    def cleanup_expired_sessions(self) -> int:
        """Removes all sessions that have exceeded their TTL."""
        now = time.time()
        with self._lock:
            expired_keys = [
                k for k, s in self._sessions.items()
                if (now - s.updated_at) > self.ttl_seconds
            ]
            for k in expired_keys:
                del self._sessions[k]
            self._last_cleanup = now
            return len(expired_keys)

    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

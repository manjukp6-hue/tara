"""
python/tara_core/event_bus.py

TARA AI Event Bus & Telemetry Broker for TARA Core.
Provides decoupled, asynchronous and synchronous event distribution across all cognitive layers:
Perception, World Model, Reasoning, Planning, Execution, Memory, Knowledge, Learning, Security, and Hardware.

Architectural Guarantees:
1. Thread-safe subscription and event dispatching.
2. Wildcard topic filtering (e.g., 'security.*', 'cognitive.*', '*').
3. Bounded event journaling (circular buffer) preventing unbounded memory growth.
4. Fail-safe isolation: subscriber exceptions never crash the event publisher or the Brain.
5. Zero external broker dependencies (pure Python local TARA AI implementation).
"""

import time
import fnmatch
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque

logger = logging.getLogger("TARA.EventBus")


@dataclass
class TaraEvent:
    event_type: str
    payload: Dict[str, Any]
    source: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: f"evt_{int(time.time()*1000)}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "source": self.source,
            "timestamp": self.timestamp
        }

    @property
    def topic(self) -> str:
        return self.event_type


class TaraEventBus:
    """
    Central event dispatch and pub-sub hub connecting all TARA subsystems.
    """
    _instance: Optional["TaraEventBus"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._subscribers: Dict[str, List[Callable[[TaraEvent], None]]] = {}
        self._history: deque = deque(maxlen=max_history)
        self._stats: Dict[str, int] = {
            "published_count": 0,
            "dispatched_count": 0,
            "error_count": 0
        }
        self._bus_lock = threading.RLock()

    @classmethod
    def get_default(cls, max_history: int = 1000) -> "TaraEventBus":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(max_history=max_history)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def subscribe(self, pattern: str, callback: Callable[[TaraEvent], None]) -> str:
        """
        Subscribes a callback to an event type or wildcard pattern (e.g. 'security.*', 'cognitive.*', '*').
        """
        with self._bus_lock:
            if pattern not in self._subscribers:
                self._subscribers[pattern] = []
            if callback not in self._subscribers[pattern]:
                self._subscribers[pattern].append(callback)
            return pattern

    def unsubscribe(self, pattern: str, callback: Callable[[TaraEvent], None]) -> bool:
        """Removes a callback subscription."""
        with self._bus_lock:
            if pattern in self._subscribers and callback in self._subscribers[pattern]:
                self._subscribers[pattern].remove(callback)
                if not self._subscribers[pattern]:
                    del self._subscribers[pattern]
                return True
            return False

    def publish(self, event_type: str, payload: Optional[Dict[str, Any]] = None, source: str = "TARA_CORE") -> TaraEvent:
        """
        Publishes an event to all matching subscribers synchronously.
        Isolates exceptions so failing callbacks do not affect the publisher.
        """
        event = TaraEvent(
            event_type=event_type,
            payload=payload or {},
            source=source
        )

        with self._bus_lock:
            self._history.append(event)
            self._stats["published_count"] += 1

            matched_callbacks: Set[Callable[[TaraEvent], None]] = set()
            for pattern, callbacks in self._subscribers.items():
                if pattern == "*" or pattern == event_type or fnmatch.fnmatch(event_type, pattern):
                    matched_callbacks.update(callbacks)

            for cb in matched_callbacks:
                try:
                    cb(event)
                    self._stats["dispatched_count"] += 1
                except Exception as ex:
                    self._stats["error_count"] += 1
                    logger.warning(f"Error in EventBus subscriber callback for '{event_type}': {ex}")

        return event

    def get_history(self, event_type_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent event history matching an optional filter."""
        with self._bus_lock:
            results = []
            for ev in reversed(self._history):
                if event_type_filter is None or fnmatch.fnmatch(ev.event_type, event_type_filter):
                    results.append(ev.to_dict())
                if len(results) >= limit:
                    break
            return results

    def get_stats(self) -> Dict[str, Any]:
        with self._bus_lock:
            return dict(self._stats)

    def clear_history(self) -> None:
        with self._bus_lock:
            self._history.clear()


# Backward compatibility aliases

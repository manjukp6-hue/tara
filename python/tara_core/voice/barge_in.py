"""
python/tara_core/voice/barge_in.py

Barge-In and Interruption State Coordinator for TARA.

Features:
- Thread-safe tracking of TARA speaking state and turn tokens.
- User speech start detection for immediate playback cutoff.
- Interruption metrics tracking.
- Prevents audio collisions between user speech and TARA speech.
"""

import threading
import time
from typing import Dict, Any, Optional


class BargeInCoordinator:
    """
    Coordinates interruption and barge-in state during live voice-to-voice dialogues.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._is_speaking: bool = False
        self._current_turn_id: Optional[str] = None
        self._last_speech_time: float = 0.0
        self._interruption_count: int = 0
        self._interrupted_turn_id: Optional[str] = None

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    @property
    def interruption_count(self) -> int:
        with self._lock:
            return self._interruption_count

    def start_speaking(self, turn_id: str) -> None:
        """Notifies the coordinator that TARA has started outputting speech."""
        with self._lock:
            self._is_speaking = True
            self._current_turn_id = turn_id
            self._last_speech_time = time.time()

    def stop_speaking(self, turn_id: Optional[str] = None) -> None:
        """Notifies the coordinator that TARA speech playback ended."""
        with self._lock:
            if turn_id is None or self._current_turn_id == turn_id:
                self._is_speaking = False
                self._current_turn_id = None

    def handle_user_barge_in(self) -> Dict[str, Any]:
        """
        Invoked when user speech or mic input is detected.
        If TARA is currently speaking, triggers immediate cutoff.
        """
        with self._lock:
            was_speaking = self._is_speaking
            cut_turn = self._current_turn_id
            if was_speaking:
                self._is_speaking = False
                self._interruption_count += 1
                self._interrupted_turn_id = cut_turn
                self._current_turn_id = None

            return {
                "interrupted": was_speaking,
                "interrupted_turn_id": cut_turn,
                "timestamp": time.time(),
                "total_interruptions": self._interruption_count
            }

    def should_abort_synthesis(self, turn_id: str) -> bool:
        """Checks if a synthesis or streaming task has been interrupted."""
        with self._lock:
            if not self._is_speaking:
                return True
            return self._current_turn_id != turn_id

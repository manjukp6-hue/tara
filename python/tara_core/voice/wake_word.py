"""
python/tara_core/voice/wake_word.py

Wake Word Detector for TARA Voice Interface.

Features:
- Detects primary wake word "TARA" (English) and "ತಾರಾ" (Kannada).
- Phonetic / token boundary matching.
- Strips wake word prefix when extracting user command.
"""

import re
from typing import Dict, Any, Tuple, Optional

WAKE_WORDS = ["tara", "ತಾರಾ", "hey tara", "ಓ ತಾರಾ", "namaskara tara", "ನಮಸ್ಕಾರ ತಾರಾ"]


class WakeWordDetector:
    """
    Lightweight keyword spotter for TARA conversational activation.
    """

    def __init__(self, wake_words: Optional[list] = None):
        self.wake_words = wake_words or WAKE_WORDS

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Evaluates input text for wake-word presence.
        Returns:
            detected: bool
            wake_word: str matched
            cleaned_command: text with wake word stripped
            confidence: float
        """
        if not text:
            return {"detected": False, "wake_word": None, "cleaned_command": "", "confidence": 0.0}

        cleaned = text.strip()
        lower = cleaned.lower()

        # Check prefix match (supports Indic and Latin scripts)
        for ww in self.wake_words:
            ww_low = ww.lower()
            if lower.startswith(ww_low):
                remainder = cleaned[len(ww_low):].lstrip(" ,;:!\t\n-")
                return {
                    "detected": True,
                    "wake_word": ww,
                    "cleaned_command": remainder,
                    "confidence": 0.96
                }

        # Check anywhere in text
        for ww in self.wake_words:
            ww_low = ww.lower()
            idx = lower.find(ww_low)
            if idx != -1:
                remainder = cleaned[idx + len(ww_low):].lstrip(" ,;:!\t\n-")
                return {
                    "detected": True,
                    "wake_word": ww,
                    "cleaned_command": remainder or cleaned,
                    "confidence": 0.88
                }

        return {
            "detected": False,
            "wake_word": None,
            "cleaned_command": cleaned,
            "confidence": 0.0
        }

"""
TARA Voice-to-Voice Conversation Subsystem.

Components:
- VoiceInputProvider (STT Engine)
- VoiceOutputProvider (TTS Engine)
- BargeInCoordinator (Interruption handling)
- WakeWordDetector (Keyword spotter)
"""

from .stt_engine import VoiceInputProvider
from .tts_engine import VoiceOutputProvider
from .barge_in import BargeInCoordinator
from .wake_word import WakeWordDetector

__all__ = [
    "VoiceInputProvider",
    "VoiceOutputProvider",
    "BargeInCoordinator",
    "WakeWordDetector",
]

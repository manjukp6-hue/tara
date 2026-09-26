"""
python/tara_core/language_registry.py

Dynamic, Open-Ended Language Registry for TARA.
Decouples language support from fixed inventories or hardcoded counts.
Enables adding new languages, scripts, glossaries, and prompt templates dynamically
without modifying core brain logic or retraining requirements.
"""

import re
import logging
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("TARA.LanguageRegistry")


@dataclass
class LanguageDefinition:
    code: str                  # ISO 639-1 / 639-2 or custom (e.g., 'kn', 'hi', 'en', 'fr')
    name: str                  # English display name (e.g., 'Kannada', 'Hindi')
    native_name: str           # Autonym (e.g., 'ಕನ್ನಡ', 'हिन्दी')
    script: str                # Script family (e.g., 'Kannada', 'Devanagari', 'Latin')
    sample_greetings: List[str] = field(default_factory=list)
    stop_words: List[str] = field(default_factory=list)
    system_prompt_adaptation: str = ""
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "native_name": self.native_name,
            "script": self.script,
            "sample_greetings": self.sample_greetings,
            "stop_words": self.stop_words,
            "system_prompt_adaptation": self.system_prompt_adaptation,
            "enabled": self.enabled,
            "metadata": self.metadata,
            "registered_at": self.registered_at
        }


class LanguageRegistry:
    """
    Central registry for language extensibility in TARA.
    Allows registering any natural or programming language dynamically.
    """
    _instance: Optional["LanguageRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self._languages: Dict[str, LanguageDefinition] = {}
        self._lock = threading.RLock()
        self._seed_initial_languages()

    @classmethod
    def get_default(cls) -> "LanguageRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def register_language(self, lang: LanguageDefinition) -> bool:
        if not lang.code or not lang.name:
            raise ValueError("Language definition must include valid code and name.")
        code = lang.code.lower().strip()
        with self._lock:
            self._languages[code] = lang
            logger.info(f"Registered language '{lang.name}' ({code})")
            return True

    def unregister_language(self, code: str) -> bool:
        code = code.lower().strip()
        with self._lock:
            if code in self._languages:
                del self._languages[code]
                logger.info(f"Unregistered language '{code}'")
                return True
            return False

    def get_language(self, code: str) -> Optional[LanguageDefinition]:
        with self._lock:
            return self._languages.get(code.lower().strip())

    def list_languages(self, enabled_only: bool = True) -> List[LanguageDefinition]:
        with self._lock:
            if enabled_only:
                return [l for l in self._languages.values() if l.enabled]
            return list(self._languages.values())

    def detect_language(self, text: str) -> Optional[str]:
        """
        Lightweight script/vocabulary detection based on unicode ranges and greetings.
        """
        with self._lock:
            # Check script unicode blocks
            for code, lang in self._languages.items():
                if not lang.enabled:
                    continue
                # Check sample greetings
                for greet in lang.sample_greetings:
                    if greet.lower() in text.lower():
                        return code

            # Check Kannada script range \u0C80-\u0CFF
            if re.search(r"[\u0C80-\u0CFF]", text):
                return "kn"
            # Devanagari \u0900-\u097F
            if re.search(r"[\u0900-\u097F]", text):
                return "hi"
            # Tamil \u0B80-\u0BFF
            if re.search(r"[\u0B80-\u0BFF]", text):
                return "ta"
            # Telugu \u0C00-\u0C7F
            if re.search(r"[\u0C00-\u0C7F]", text):
                return "te"

            return "en"

    def count(self) -> int:
        with self._lock:
            return len(self._languages)

    def _seed_initial_languages(self) -> None:
        """Seed initial baseline languages without imposing maximum limits."""
        seeds = [
            LanguageDefinition(code="en", name="English", native_name="English", script="Latin", sample_greetings=["hello", "hi", "good morning"]),
            LanguageDefinition(code="kn", name="Kannada", native_name="ಕನ್ನಡ", script="Kannada", sample_greetings=["ನಮಸ್ಕಾರ", "ಶುಭೋದಯ"]),
            LanguageDefinition(code="hi", name="Hindi", native_name="हिन्दी", script="Devanagari", sample_greetings=["नमस्ते", "नमस्कार"]),
            LanguageDefinition(code="te", name="Telugu", native_name="తెలుగు", script="Telugu", sample_greetings=["నమస్కారం"]),
            LanguageDefinition(code="ta", name="Tamil", native_name="தமிழ்", script="Tamil", sample_greetings=["வணக்கம்"]),
            LanguageDefinition(code="ml", name="Malayalam", native_name="മലയാളം", script="Malayalam", sample_greetings=["നമസ്കാരം"]),
            LanguageDefinition(code="mr", name="Marathi", native_name="मराठी", script="Devanagari", sample_greetings=["नमस्कार"]),
            LanguageDefinition(code="bn", name="Bengali", native_name="বাংলা", script="Bengali", sample_greetings=["নমস্কার"]),
            LanguageDefinition(code="gu", name="Gujarati", native_name="ગુજરાતી", script="Gujarati", sample_greetings=["નમસ્તે"]),
            LanguageDefinition(code="pa", name="Punjabi", native_name="ਪੰਜਾਬੀ", script="Gurmukhi", sample_greetings=["ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ"]),
            LanguageDefinition(code="or", name="Odia", native_name="ଓଡ଼ିଆ", script="Odia", sample_greetings=["ନମସ୍କାର"]),
            LanguageDefinition(code="as", name="Assamese", native_name="অসমীয়া", script="Bengali", sample_greetings=["নমস্কাৰ"]),
            LanguageDefinition(code="ur", name="Urdu", native_name="اردو", script="Arabic", sample_greetings=["سلام", "آداب"]),
            LanguageDefinition(code="sa", name="Sanskrit", native_name="संस्कृतम्", script="Devanagari", sample_greetings=["नमस्ते"]),
            LanguageDefinition(code="ks", name="Kashmiri", native_name="کٲشُر", script="Arabic", sample_greetings=["سلام"]),
            LanguageDefinition(code="sd", name="Sindhi", native_name="سنڌي", script="Arabic", sample_greetings=["سلام"]),
            LanguageDefinition(code="ne", name="Nepali", native_name="नेपाली", script="Devanagari", sample_greetings=["नमस्ते"]),
            LanguageDefinition(code="kok", name="Konkani", native_name="कोंकणी", script="Devanagari", sample_greetings=["नमस्कार"]),
            LanguageDefinition(code="mai", name="Maithili", native_name="मैथिली", script="Devanagari", sample_greetings=["प्रणाम"]),
            LanguageDefinition(code="bho", name="Bhojpuri", native_name="भोजपुरी", script="Devanagari", sample_greetings=["प्रणाम"])
        ]
        for s in seeds:
            self._languages[s.code] = s

"""
TARA/RULES/nlu/language_detector.py

Language detection for TARA Natural-Language Rulebook.
Supports English, Kannada (script), Kanglish (phonetic Kannada in Latin script), and Hindi.
"""

import re
from typing import Dict, Any

# Unicode ranges
KANNADA_RANGE = re.compile(r'[ಀ-೿]')
DEVANAGARI_RANGE = re.compile(r'[ऀ-ॿ]')

# Common Kanglish morphological markers, keywords, and auxiliary verb patterns
KANGLISH_PATTERNS = [
    r'\bmadbeda\b', r'\btorisbedi\b', r'\bkelbeku\b', r'\bkeli\b', r'\bkelu\b',
    r'\bmathadu\b', r'\bmathadovaga\b', r'\bkannadadalli\b', r'\bkannada\b',
    r'\bsarkari\b', r'\bniyama\b', r'\bniyamagalu\b', r'\bpalisi\b', r'\bmadi\b',
    r'\bmunche\b', r'\bnanna\b', r'\bnannu\b', r'\bhelidange\b', r'\bbeda\b',
    r'\bbedve\b', r'\bkodbarda\b', r'\billade\b', r'\bbadalavane\b', r'\bmukhyavada\b',
    r'\byavaga\b', r'\balla\b', r'\bhage\b', r'\bhesaru\b', r'\bmadodu\b'
]

class LanguageDetector:
    """Detects rule language: en, kn, kanglish, hi."""
    
    @staticmethod
    def detect_language(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            return {"language": "en", "confidence": 1.0}
        
        # Check Kannada script
        kn_matches = len(KANNADA_RANGE.findall(cleaned))
        if kn_matches > 0:
            return {"language": "kn", "confidence": min(1.0, kn_matches / max(1, len(cleaned.split())))}
            
        # Check Hindi / Devanagari script
        hi_matches = len(DEVANAGARI_RANGE.findall(cleaned))
        if hi_matches > 0:
            return {"language": "hi", "confidence": min(1.0, hi_matches / max(1, len(cleaned.split())))}
            
        # Check Kanglish (phonetic Kannada written in Latin alphabet)
        lower_text = cleaned.lower()
        kanglish_hits = 0
        for pat in KANGLISH_PATTERNS:
            if re.search(pat, lower_text):
                kanglish_hits += 1
                
        if kanglish_hits >= 1:
            confidence = min(1.0, 0.5 + (kanglish_hits * 0.2))
            return {"language": "kanglish", "confidence": confidence, "detected_markers": kanglish_hits}
            
        # Default to English
        return {"language": "en", "confidence": 0.95}

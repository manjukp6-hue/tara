"""
TARA/RULES/nlu/rule_parser.py

Parses freeform notebook lines into structured rule candidate representations.
"""

import re
from typing import List, Dict, Any
from .language_detector import LanguageDetector
from .semantic_normalizer import SemanticNormalizer

class RuleParser:
    """Parses raw text into rule candidate objects."""

    @staticmethod
    def parse_text(content: str) -> List[Dict[str, Any]]:
        """
        Parses notebook-style text into a list of parsed rule candidates.
        Strips header banners, empty lines, comments, and numbering.
        """
        candidates: List[Dict[str, Any]] = []
        lines = content.splitlines()
        
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            # Ignore notebook headers or comments
            if line.upper() in ["RULE BOOK", "RULEBOOK", "RULES", "TARA RULES"]:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue
                
            # Strip numeric or bullet prefix: e.g. "1. ", "2) ", "- ", "* "
            cleaned = re.sub(r'^(\d+[\.\)]|\-|\*)\s*', '', line).strip()
            if not cleaned:
                continue
                
            # Detect language
            lang_info = LanguageDetector.detect_language(cleaned)
            lang = lang_info["language"]
            
            # Semantic normalization
            normalized = SemanticNormalizer.normalize_rule(cleaned, lang)
            normalized["line_number"] = len(candidates) + 1
            candidates.append(normalized)
            
        return candidates

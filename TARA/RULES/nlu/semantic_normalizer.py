"""
TARA/RULES/nlu/semantic_normalizer.py

Semantic interpretation & normalization of natural-language rules.
Maps natural language in English, Kannada, Kanglish to canonical intents, actions, and entities.
"""

import re
from typing import Dict, Any, List, Optional

class SemanticNormalizer:
    """Normalizes natural-language rule text to canonical policy semantics."""

    @staticmethod
    def normalize_rule(text: str, language: str) -> Dict[str, Any]:
        lower = text.lower().strip()
        
        # 1. Prohibitive / Negative patterns ("madbeda", "do not", "never", "ತೋರಿಸಬೇಡಿ", "ಮಾಡಬೇಡಿ")
        is_prohibitive = False
        prohibit_patterns = [
            r'\bdo not\b', r"don't", r'\bnever\b', r'\bmust not\b', r'\bprohibit\b',
            r'\bmadbeda\b', r'\btorisbedi\b', r'\bbeda\b', r'\bbedve\b', r'\bkodbarda\b',
            r'ತೋರಿಸಬೇಡಿ', r'ಮಾಡಬೇಡಿ', r'ಬಾರದು', r'ಬೇಡ', r'ತಿರಸ್ಕರಿಸಿ', r'\bstop\b', r'\bblock\b'
        ]
        for p in prohibit_patterns:
            if re.search(p, lower):
                is_prohibitive = True
                break
                
        # 2. Confirmation / Approval patterns ("ask me before", "kelbeku", "ಮುಂಚೆ ನನ್ನ ಕೇಳಿ")
        requires_confirmation = False
        confirm_patterns = [
            r'\bask me before\b', r'\bask before\b', r'\bwith my approval\b', r'\bwith approval\b',
            r'\bseek approval\b', r'\brequire confirmation\b', r'\bwithout my approval\b',
            r'\bkelbeku\b', r'\bkeli\b', r'\bkelu\b', r'\bmunche nanna keli\b',
            r'ಕೇಳಿ', r'ಅನುಮೋದನೆ', r'ಅನುಮತಿ ಪಡೆಯಿರಿ'
        ]
        for p in confirm_patterns:
            if re.search(p, lower):
                requires_confirmation = True
                break
                
        # 3. Ambiguous blanket power / bypass patterns ("do what i say", "nannu helidange madu")
        is_ambiguous = False
        ambiguous_patterns = [
            r'\bdo what i say\b', r'\bdo whatever i say\b', r'\beverything i say\b',
            r'\beverything i tell you\b', r'\bdo anything\b', r'\bignore (all|safety|rules)\b',
            r'\bbypass (rules|safety)\b', r'\bnannu helidange madu\b', r'\bnanu helidange madu\b',
            r'ನಾನು ಹೇಳಿದಂತೆ ಮಾಡು', r'ನಾನು ಹೇಳಿದ ಎಲ್ಲವನ್ನೂ ಮಾಡು'
        ]
        for p in ambiguous_patterns:
            if re.search(p, lower):
                is_ambiguous = True
                break

        # 4. Domain & Entity Recognition
        category = "CREATOR_CUSTOM"
        canonical_meaning = text
        action = "ENFORCE"
        conditions = {}
        is_mandatory_candidate = False

        # Law & Government Rules
        if any(w in lower for w in ["government rules", "country and government", "applicable country", "applicable law", 
                                    "statutory", "sarkari", "niyama", "ಸರ್ಕಾರ", "ಕಾನೂನು"]):
            category = "MANDATORY_LAW"
            action = "ENFORCE"
            canonical_meaning = "Follow and protect applicable country and government laws and rules."
            conditions = {"domain": "statutory_compliance", "scope": "global"}
            is_mandatory_candidate = True

        # Child Safety / CSAM
        elif any(w in lower for w in ["child sexual abuse", "child pornography", "csam", "child exploitation", 
                                      "ಮಕ್ಕಳ ಲೈಂಗಿಕ"]):
            category = "CHILD_SAFETY"
            action = "DENY"
            canonical_meaning = "Do not show, generate, distribute, or assist with child sexual abuse material or child pornography."
            conditions = {"domain": "child_safety", "content_type": ["csam", "child_abuse"]}
            is_mandatory_candidate = True

        # Adult / Nude / Sexual Content
        elif any(w in lower for w in ["nude photos", "nude videos", "nude", "sexual content", "naked", "porn",
                                      "ನಗ್ನ", "ಕಾಮ"]):
            category = "CONTENT_SAFETY"
            action = "DENY" if is_prohibitive else "DENY"
            canonical_meaning = "Do not show, generate, or distribute nude photos, nude videos, or explicit sexual content."
            conditions = {"domain": "content_safety", "media": ["nude_photo", "nude_video", "explicit_content"]}
            is_mandatory_candidate = True

        # Creator Instructions
        elif any(w in lower for w in ["authorized creator instructions", "creator instructions", "creator order", "ಆಜ್ಞೆ"]):
            category = "CREATOR_AUTHORITY"
            action = "ENFORCE"
            canonical_meaning = "Follow authorized Creator instructions."
            conditions = {"domain": "creator_authority"}

        # File Deletion Protection
        elif any(w in lower for w in ["deleting an important file", "delete an important file", "important file",
                                      "delete madoke", "delete madodu", "ಅಳಿಸುವ ಮೊದಲು"]):
            category = "DATA_MUTATION"
            if is_prohibitive and any(w in lower for w in ["never ask", "do not ask", "don't ask", "kelbeda", "ಕೇಳಬೇಡಿ"]):
                action = "ALLOW"
                canonical_meaning = "Do not ask Creator before deleting files."
            else:
                action = "REQUIRE_CREATOR_CONFIRMATION"
                canonical_meaning = "Always ask Creator for confirmation before deleting an important file."
            conditions = {"action_type": "file_deletion", "critical_target": True}

        # Language Preference (English)
        elif any(w in lower for w in ["use english", "speak english", "english only", "always use english"]):
            category = "LANGUAGE_PREFERENCE"
            action = "ENFORCE_SETTING"
            canonical_meaning = "Use English language."
            conditions = {"target_setting": "language", "preferred_language": "en"}

        # Language Preference (Kannada)
        elif any(w in lower for w in ["use kannada", "speak kannada", "when i speak kannada", "kannadadalli mathadu",
                                      "kannada mathadu", "ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡಿ", "ಕನ್ನಡವನ್ನು ಬಳಸಿ"]):
            category = "LANGUAGE_PREFERENCE"
            action = "ENFORCE_SETTING"
            canonical_meaning = "Use Kannada when Creator communicates in Kannada."
            conditions = {"target_setting": "language", "preferred_language": "kn"}

        # Machine Configuration Protection
        elif any(w in lower for w in ["modify my machine configuration", "machine configuration", "system configuration",
                                      "configuration change", "ಕಾನ್ಫಿಗರೇಶನ್"]):
            category = "SYSTEM_INTEGRITY"
            action = "REQUIRE_CREATOR_CONFIRMATION"
            canonical_meaning = "Do not modify machine configuration without explicit Creator approval."
            conditions = {"action_type": "system_config_mutation"}

        # Assistant Name Preference
        elif any(w in lower for w in ["preferred assistant name is tara", "assistant name is tara", "name is tara", "ಹೆಸರು ತಾರಾ"]):
            category = "CREATOR_CUSTOM"
            action = "ENFORCE_SETTING"
            canonical_meaning = "Maintain assistant name preference as TARA."
            conditions = {"target_setting": "assistant_name", "value": "TARA"}

        # Privacy, Keys & Recovery Protection
        elif any(w in lower for w in ["private key", "seed phrase", "recovery secret", "credentials", "ಖಾಸಗಿ ಕೀ", "ರಹಸ್ಯ"]):
            category = "CREDENTIAL_PROTECTION"
            action = "DENY"
            canonical_meaning = "Protect private keys, recovery secrets, and credentials against disclosure."
            conditions = {"domain": "credential_protection"}
            is_mandatory_candidate = True

        # Anti-privilege escalation
        elif any(w in lower for w in ["privilege escalation", "permission bypass", "self-elevation"]):
            category = "SECURITY"
            action = "DENY"
            canonical_meaning = "Block unauthorized privilege escalation and permission bypass."
            conditions = {"domain": "privilege_escalation"}
            is_mandatory_candidate = True

        # Handle ambiguous blanket rules
        if is_ambiguous:
            action = "AMBIGUOUS"
            category = "AMBIGUOUS"
            canonical_meaning = f"Ambiguous instruction requires clarification: '{text}'"

        return {
            "original_text": text,
            "language": language,
            "category": category,
            "action": action,
            "canonical_meaning": canonical_meaning,
            "conditions": conditions,
            "is_mandatory": is_mandatory_candidate,
            "is_ambiguous": is_ambiguous
        }

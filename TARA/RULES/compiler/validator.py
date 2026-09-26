"""
TARA/RULES/compiler/validator.py

Policy Safety Validator.
Inspects parsed rule candidates for:
- Ambiguity
- Privilege escalation attempts
- Attempts to disable security
- Attempts to override mandatory baseline protections
"""

from typing import List, Dict, Any, Tuple
from .policy_schema import PolicyPriority, RuleAction, RuleCategory

class PolicyValidator:
    """Validates rules against mandatory invariants and safety boundaries."""

    MANDATORY_CATEGORIES = {
        RuleCategory.MANDATORY_LAW.value,
        RuleCategory.CHILD_SAFETY.value,
        RuleCategory.CONTENT_SAFETY.value,
        RuleCategory.SECURITY.value,
        RuleCategory.CREDENTIAL_PROTECTION.value,
        RuleCategory.PRIVACY.value
    }

    @classmethod
    def validate_rule_candidate(cls, candidate: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates a single parsed rule candidate.
        Returns: (is_valid, validation_message, sanitized_candidate)
        """
        text_lower = candidate.get("original_text", "").lower()
        
        # 1. Detect attempts to override mandatory protections
        override_law = any(w in text_lower for w in ["ignore law", "ignore government", "bypass government", "allow csam", 
                                                      "allow child porn", "allow nude", "permit nudity", "disable safety"])
        if override_law:
            candidate["action"] = RuleAction.REJECTED.value
            candidate["status"] = "REJECTED"
            return False, "RULE_REJECTED: Attempt to override mandatory law or safety baseline", candidate

        # 2. Detect privilege escalation attempts
        escalate = any(w in text_lower for w in ["grant me root", "become creator", "grant admin", "override creator",
                                                 "bypass creator", "disable lock", "disable lockdown"])
        if escalate:
            candidate["action"] = RuleAction.REJECTED.value
            candidate["status"] = "REJECTED"
            return False, "RULE_REJECTED: Unauthorized privilege escalation or security disablement attempt", candidate

        # 3. Detect Ambiguous blanket power statements
        if candidate.get("is_ambiguous"):
            candidate["action"] = RuleAction.AMBIGUOUS.value
            candidate["status"] = "AMBIGUOUS"
            return True, "RULE_AMBIGUOUS: Instruction requires Creator clarification before activation", candidate

        # 4. Valid normal rule
        candidate["status"] = "ACTIVE"
        return True, "RULE_VALID", candidate

    @classmethod
    def verify_baseline_integrity(cls, mandatory_rules: List[Dict[str, Any]], active_rules: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Ensures all mandatory baseline protections are preserved in active_rules.
        Mandatory rules CANNOT be dropped or removed.
        """
        missing = []
        active_meanings = {r.get("meaning") or r.get("canonical_meaning") for r in active_rules}
        
        for m in mandatory_rules:
            m_meaning = m.get("meaning") or m.get("canonical_meaning")
            if m_meaning and m_meaning not in active_meanings:
                missing.append(f"Missing mandatory baseline rule: {m_meaning}")
                
        return (len(missing) == 0), missing

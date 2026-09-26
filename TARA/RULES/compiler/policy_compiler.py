"""
TARA/RULES/compiler/policy_compiler.py

Main Policy Compiler.
Compiles RULEBOOK.txt + DEFAULT_SAFE_RULES.txt into a verified CompiledPolicy object.
"""

import os
import time
from typing import List, Dict, Any, Tuple, Optional
from ..nlu.rule_parser import RuleParser
from .policy_schema import CompiledPolicy, Rule, PolicyPriority, RuleAction, RuleCategory
from .validator import PolicyValidator
from .conflict_resolver import ConflictResolver

CANONICAL_CREATOR_ID = "ROOT_OPERATOR"
DEFAULT_DISPLAY_NAME = "OPERATOR_ROOT"

class PolicyCompiler:
    """Compiles raw rulebook text into structured, cryptographically signable policies."""

    def __init__(self, creator_id: str = CANONICAL_CREATOR_ID, display_name: str = DEFAULT_DISPLAY_NAME):
        self.creator_id = creator_id
        self.display_name = display_name

    def compile_rules(
        self,
        rulebook_text: str,
        default_safe_text: str,
        target_version: int = 1,
        signing_key_version: int = 1
    ) -> Tuple[CompiledPolicy, List[str], List[Dict[str, Any]]]:
        """
        Compiles rulebook text and baseline safe text.
        Returns: (compiled_policy, validation_errors, conflict_reports)
        """
        # 1. Parse Baseline Mandatory Safe Rules
        baseline_candidates = RuleParser.parse_text(default_safe_text)
        mandatory_rules_data: List[Dict[str, Any]] = []
        for idx, m in enumerate(baseline_candidates, start=1):
            m["rule_id"] = f"RULE-BASE-{idx:03d}"
            m["version"] = target_version
            m["priority"] = PolicyPriority.MANDATORY_LAW_SAFETY.value if m["category"] in [
                RuleCategory.MANDATORY_LAW.value, RuleCategory.CHILD_SAFETY.value, RuleCategory.CONTENT_SAFETY.value
            ] else PolicyPriority.TARA_SECURITY.value
            m["scope"] = "GLOBAL"
            m["is_mandatory"] = True
            m["meaning"] = m["canonical_meaning"]
            m["status"] = "ACTIVE"
            m["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            m["updated_at"] = m["created_at"]
            mandatory_rules_data.append(m)

        # 2. Parse Creator Custom Rules
        creator_candidates = RuleParser.parse_text(rulebook_text)
        creator_rules_data: List[Dict[str, Any]] = []
        validation_errors: List[str] = []

        for idx, c in enumerate(creator_candidates, start=1):
            is_valid, msg, sanitized = PolicyValidator.validate_rule_candidate(c)
            if not is_valid:
                validation_errors.append(f"Line {c.get('line_number')}: {msg} ('{c.get('original_text')}')")
                continue
                
            sanitized["rule_id"] = f"RULE-CR-{idx:03d}"
            sanitized["version"] = target_version
            sanitized["priority"] = PolicyPriority.CREATOR_RULE.value
            sanitized["scope"] = "GLOBAL"
            sanitized["meaning"] = sanitized["canonical_meaning"]
            sanitized["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            sanitized["updated_at"] = sanitized["created_at"]
            creator_rules_data.append(sanitized)

        # 3. Verify Mandatory Baseline Preservation
        # Baseline rules cannot be dropped or replaced by creator rules
        all_rules_data = list(mandatory_rules_data) + creator_rules_data
        
        # 4. Resolve Conflicts across rules
        resolved_rules, conflicts = ConflictResolver.resolve_conflicts(all_rules_data)

        # 5. Convert to Rule Objects
        compiled_rule_objects: List[Rule] = []
        mandatory_count = 0
        creator_count = 0
        ambiguous_count = 0

        for r in resolved_rules:
            rule_obj = Rule(
                rule_id=r["rule_id"],
                version=r.get("version", target_version),
                category=r["category"],
                meaning=r.get("meaning") or r.get("canonical_meaning", ""),
                priority=r.get("priority", PolicyPriority.CREATOR_RULE.value),
                scope=r.get("scope", "GLOBAL"),
                action=r["action"],
                conditions=r.get("conditions", {}),
                status=r.get("status", "ACTIVE"),
                original_text=r["original_text"],
                language=r["language"],
                is_mandatory=r.get("is_mandatory", False),
                created_at=r.get("created_at", ""),
                updated_at=r.get("updated_at", "")
            )
            if rule_obj.is_mandatory:
                mandatory_count += 1
            else:
                creator_count += 1
            if rule_obj.action == RuleAction.AMBIGUOUS.value:
                ambiguous_count += 1
            compiled_rule_objects.append(rule_obj)

        policy = CompiledPolicy(
            policy_version=target_version,
            creator_id=self.creator_id,
            display_name=self.display_name,
            rules=compiled_rule_objects,
            mandatory_rule_count=mandatory_count,
            creator_rule_count=creator_count,
            ambiguous_rule_count=ambiguous_count,
            compiled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            signing_key_version=signing_key_version
        )
        policy.digest_sha256 = policy.compute_digest()

        return policy, validation_errors, conflicts

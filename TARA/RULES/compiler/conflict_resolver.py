"""
TARA/RULES/compiler/conflict_resolver.py

Conflict detection & resolution for TARA policies.
Enforces 4-tier hierarchy:
MANDATORY LAW / SAFETY (Priority 1)
  ↓
TARA SECURITY (Priority 2)
  ↓
CREATOR RULES (Priority 3)
  ↓
USER REQUESTS (Priority 4)

Between Creator rules:
- Detects contradictions
- Resolves by explicit priority if set
- Otherwise marks CONFLICTING and asks Creator for clarification. Never silently assumes.
"""

from typing import List, Dict, Any, Tuple
from .policy_schema import PolicyPriority, RuleAction

class ConflictResolver:
    """Detects and resolves rule conflicts."""

    @classmethod
    def resolve_conflicts(cls, rules: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Resolves conflicts across rules.
        Returns (resolved_rules, conflict_reports).
        """
        resolved: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = []

        # Index rules by domain/topic conditions
        topic_map: Dict[str, List[Dict[str, Any]]] = {}
        for r in rules:
            conds = r.get("conditions", {})
            domain = conds.get("domain") or conds.get("target_setting") or conds.get("action_type") or r.get("category")
            if domain:
                topic_map.setdefault(domain, []).append(r)
            else:
                resolved.append(r)

        for domain, group in topic_map.items():
            if len(group) == 1:
                resolved.append(group[0])
                continue

            # Group has multiple rules on the same domain
            # Sort by priority (1 is highest priority)
            sorted_group = sorted(group, key=lambda x: x.get("priority", 3))
            
            top_rule = sorted_group[0]
            top_priority = top_rule.get("priority", 3)
            
            # Check for intra-level conflict at the highest priority
            same_priority = [r for r in sorted_group if r.get("priority", 3) == top_priority]
            
            if len(same_priority) > 1:
                # Check if actions/conditions contradict
                actions = {r.get("action") for r in same_priority}
                settings = {str(r.get("conditions")) for r in same_priority}
                if len(actions) > 1 or len(settings) > 1:
                    # Contradiction at same level!
                    conflict_report = {
                        "domain": domain,
                        "priority": top_priority,
                        "conflicting_rules": [r.get("original_text") for r in same_priority],
                        "status": "UNRESOLVED_CONFLICT",
                        "resolution_required": "Creator clarification required; dangerous interpretation rejected"
                    }
                    conflicts.append(conflict_report)
                    for r in same_priority:
                        r["status"] = "CONFLICTING"
                    resolved.extend(same_priority)
                    continue

            # Higher priority overrides lower priority
            resolved.append(top_rule)
            for lower in sorted_group[1:]:
                if lower.get("priority", 3) > top_priority:
                    lower["status"] = "SUPERSEDED_BY_HIGHER_PRIORITY"
                    # We still track it in resolved for transparency
                    resolved.append(lower)

        return resolved, conflicts
